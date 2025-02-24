import psutil
import time
from dataclasses import dataclass
import threading
import multiprocessing
import pynvml
from requests import request
import traceback


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


@dataclass
class LoadingTracker:
    task: str
    model_id: str
    # supported devices are "cuda" (gpu) and "cpu"
    device: str

    files_size_in_GB: float = 1
    model_size_in_GB: float = 1

    percent_progress_download = multiprocessing.Value("d", 0.0)
    percent_progress_load = multiprocessing.Value("d", 0.0)
    download_GB_received = multiprocessing.Value("d", 0.0)
    download_speed_MB_s = multiprocessing.Value("d", 0.0)

    available_GB = multiprocessing.Value("d", 0.0)
    peak_GB = multiprocessing.Value("d", 0.0)
    current_GB = multiprocessing.Value("d", 0.0)

    downloading_is_done = multiprocessing.Value("b", False)
    loading_is_done = multiprocessing.Value("b", False)

    logger: callable = print
    start_time = multiprocessing.Value("d", 0.0)
    end_time = multiprocessing.Value("d", 0.0)

    def mark_download_as_done(self):
        self.downloading_is_done.value = True
        self.percent_progress_download.value = 100.0
        self.set_end_time()

    def mark_load_as_done(self):
        self.loading_is_done.value = True
        self.percent_progress_load.value = 100.0
        self.set_end_time()

    def set_start_time(self):
        self.start_time.value = time.time()

    def set_end_time(self):
        self.end_time.value = time.time()

    @property
    def elapsed_time_in_seconds(self):
        if not self.loading_is_done.value:
            current = time.time()
            delta_in_seconds = current - self.start_time.value

            return round(delta_in_seconds, 2)

        delta_in_seconds = self.end_time.value - self.start_time.value

        return round(delta_in_seconds, 2)

    @timer
    def load_model_with_tracking(
        self,
        # flexibly allow for a specif endpoint to be hit to load a model, e.g. http://localhost:8002/load_model
        load_model_endpoint: str,
    ):
        self.set_start_time()

        polling_rate_s = 1

        # monitor download progress
        download_model_thread = self.monitor_downloading(
            interval_in_seconds=polling_rate_s
        )
        download_model_thread.start()

        # monitor loading progress
        monitor_process = self.monitor_loading(interval_in_seconds=polling_rate_s)
        monitor_process.start()

        # start loading the model
        load_model_thread = self.load_model(load_model_endpoint)
        load_model_thread.start()

        # we wait for model load chiefly
        load_model_thread.join()

        # if the model is loaded, these must be true
        self.mark_download_as_done()
        self.mark_load_as_done()

        self.log_percent_done_loaded()

        # wait for each step to finish
        download_model_thread.join()
        monitor_process.join()

    def load_model(self, load_model_endpoint):

        def load_model_via_http():
            max_retries = 10
            for i in range(max_retries):
                try:
                    request(method="GET", url=load_model_endpoint)
                    return
                except Exception:
                    if i >= max_retries - 1:
                        traceback.print_exc()
                    time.sleep(1)

        load_model_thread = threading.Thread(target=load_model_via_http)

        return load_model_thread

    def monitor_loading(self, interval_in_seconds: int) -> multiprocessing.Process:
        if self.device == "cuda":
            return multiprocessing.Process(
                target=self.monitor_memory_usage_cuda,
                args=[interval_in_seconds],
            )

        if self.device == "cpu":
            return multiprocessing.Process(
                target=self.monitor_memory_usage_cpu,
                args=[interval_in_seconds],
            )

        raise Exception(f"Device: {self.device} is not supported")

    def stop(self):
        self.mark_as_done()

    def monitor_downloading(
        self,
        interval_in_seconds=1,
        duration=60 * 10,
    ) -> multiprocessing.Process:
        def _monitor_downloading():
            initial_io = psutil.net_io_counters()

            prev_MB_received = 0

            for _ in range(duration // interval_in_seconds):

                # consider downloading finished if we have a significantly strong signal indicating loading is in progress
                if self.percent_progress_load.value > 2:
                    self.mark_download_as_done()
                    break

                current_io = psutil.net_io_counters()

                bytes_received = current_io.bytes_recv - initial_io.bytes_recv

                MB_received = bytes_received / (1024 * 1024)

                download_GB_received = MB_received / 1024

                chunk_of_MB_received = MB_received - prev_MB_received

                prev_MB_received = MB_received

                percent_done = (download_GB_received / self.files_size_in_GB) * 100

                # NOTE network monitoring is not exact, but pretty close to accurate
                percent_done_rounded = self.bound_and_round(
                    value=percent_done, upper_bound=100, lower_bound=0, decimals=2
                )

                download_speed_MB_s = chunk_of_MB_received / interval_in_seconds

                self.percent_progress_download.value = percent_done_rounded
                self.download_GB_received.value = download_GB_received
                self.download_speed_MB_s.value = download_speed_MB_s

                self.log_percent_done_downloaded()

                time.sleep(interval_in_seconds)

        thread = multiprocessing.Process(target=_monitor_downloading)
        return thread

    def monitor_memory_usage_cuda(
        self,
        interval_in_seconds=1,
    ):
        pynvml.nvmlInit()

        total_memory, used_memory = self.get_cuda_mem_info()

        self.available_GB.value = total_memory / (1024**3)

        initial_used_memory_in_GB = used_memory / (1024**3)

        highest_used_memory_in_GB = 0

        while True:
            if self.loading_is_done.value:
                break

            total_memory, used_memory = self.get_cuda_mem_info()

            used_memory_in_GB = used_memory / (1024**3) - initial_used_memory_in_GB

            highest_used_memory_in_GB = self.set_load_progress(
                used_memory_in_GB=used_memory_in_GB,
                highest_used_memory_in_GB=highest_used_memory_in_GB,
            )

            time.sleep(interval_in_seconds)

    def get_cuda_mem_info(self):
        current_mem_info = [
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
            for i in range(pynvml.nvmlDeviceGetCount())
        ]

        total_memory = sum([mem_info.total for mem_info in current_mem_info])
        used_memory = sum([mem_info.used for mem_info in current_mem_info])

        return total_memory, used_memory

    def monitor_memory_usage_cpu(
        self,
        interval_in_seconds=1,
    ):

        # Get initial memory usage
        initial_mem_info = psutil.virtual_memory()

        self.available_GB.value = initial_mem_info.total / (1024**3)

        initial_used_memory_in_GB = initial_mem_info.used / (1024**3)

        highest_used_memory_in_GB = 0

        while True:
            if self.loading_is_done.value:
                break
            # Get current memory usage
            current_mem_info = psutil.virtual_memory()
            used_memory_in_GB = (
                current_mem_info.used / (1024**3)
            ) - initial_used_memory_in_GB

            highest_used_memory_in_GB = self.set_load_progress(
                used_memory_in_GB=used_memory_in_GB,
                highest_used_memory_in_GB=highest_used_memory_in_GB,
            )

            time.sleep(interval_in_seconds)

    def set_load_progress(self, used_memory_in_GB, highest_used_memory_in_GB):
        self.current_GB.value = used_memory_in_GB

        highest_used_memory_in_GB = max(used_memory_in_GB, highest_used_memory_in_GB)

        self.peak_GB.value = highest_used_memory_in_GB

        percent_done = (highest_used_memory_in_GB / self.model_size_in_GB) * 100

        percent_done_rounded = self.bound_and_round(
            value=percent_done, upper_bound=100, lower_bound=0, decimals=2
        )

        self.percent_progress_load.value = percent_done_rounded

        # NOTE we stop tracking the downloader when loading gets to 2%
        # to prevent noise and potential out of order prints, we only print once we've confirmed that's done
        if self.downloading_is_done.value:
            self.log_percent_done_loaded()

        return highest_used_memory_in_GB

    def bound_and_round(
        self, value: float, upper_bound: float, lower_bound: float, decimals: int
    ):

        rounded_value = round(value, decimals)

        bounded_value = min(upper_bound, max(lower_bound, rounded_value))

        return bounded_value

    def log_percent_done_downloaded(self):
        self.logger(f"Percent downloaded: {self.percent_progress_download.value}%")

    def log_percent_done_loaded(self):
        self.logger(f"Percent loaded into memory: {self.percent_progress_load.value}%")
