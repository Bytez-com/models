import ctypes
import psutil
import time
from dataclasses import dataclass
import threading
import multiprocessing
import pynvml
from requests import request
import traceback

EXCEPTION_MAX_STRING_SIZE = 10_000


@dataclass
class LoadingTracker:
    task: str
    model_id: str
    # supported devices are "cuda" (gpu) and "cpu"
    device: str

    files_size_in_GB: float = 1
    model_size_in_GB: float = 1
    prev_dl_value: float = None
    prev_load_value: float = None

    logger: callable = print
    logging_enabled: bool = False

    def __post_init__(self):
        self.percent_progress_download = multiprocessing.Value("d", 0.0)
        self.percent_progress_load = multiprocessing.Value("d", 0.0)
        self.download_GB_received = multiprocessing.Value("d", 0.0)
        self.download_speed_MB_s = multiprocessing.Value("d", 0.0)

        self.available_GB = multiprocessing.Value("d", 0.0)
        self.peak_GB = multiprocessing.Value("d", 0.0)
        self.current_GB = multiprocessing.Value("d", 0.0)

        self.downloading_is_done = multiprocessing.Value("b", False)
        self.loading_is_done = multiprocessing.Value("b", False)
        self.loading_failed = multiprocessing.Value("b", False)
        self.loading_failed_exception = multiprocessing.Array(
            ctypes.c_char, EXCEPTION_MAX_STRING_SIZE
        )

        self.start_time = multiprocessing.Value("d", 0.0)
        self.end_time = multiprocessing.Value("d", 0.0)

    def mark_download_as_done(self):
        self.downloading_is_done.value = True
        self.percent_progress_download.value = 100.0

    def mark_load_as_done(self):
        self.loading_is_done.value = True
        self.percent_progress_load.value = 100.0

    def mark_as_failed(self):
        self.loading_is_done.value = True
        self.loading_failed.value = True

    def mark_as_done(self, failed=False, exception=""):
        if failed:
            self.mark_as_failed()

            sliced_exception = exception[:EXCEPTION_MAX_STRING_SIZE]
            self.loading_failed_exception.value = sliced_exception.encode("utf-8")
        else:
            self.mark_download_as_done()
            self.mark_load_as_done()
            self.log_percent_done_loaded()

    @property
    def elapsed_time_in_seconds(self):
        if not self.loading_is_done.value:
            current = time.time()
            delta_in_seconds = current - self.start_time.value

            return round(delta_in_seconds, 2)

        delta_in_seconds = self.end_time.value - self.start_time.value

        return round(delta_in_seconds, 2)

    def load_model_with_tracking(
        self,
        # flexibly allow for a specif endpoint to be hit to load a model, e.g. http://localhost:8002/load_model
        load_model_endpoint: str,
    ):
        start_time = time.time()

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

        elapsed_time_seconds = time.time() - start_time

        print(
            f"load_model_with_tracking took: {round(elapsed_time_seconds, 2)} seconds"
        )

        # wait for each step to finish
        download_model_thread.join()
        monitor_process.join()

    def load_model(self, load_model_endpoint):
        def load_model_via_http():
            max_retries = 10

            for i in range(max_retries):
                try:
                    result = request(method="GET", url=load_model_endpoint).json()
                    success = result["success"]
                    exception = result["exception"]
                    self.mark_as_done(failed=not success, exception=exception)
                    return

                except Exception as exception:
                    if i >= max_retries - 1:
                        traceback.print_exc()
                    time.sleep(1)
            self.mark_as_done(
                failed=True,
                exception=f"load_model_via_http timeout out after {max_retries}",
            )

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

        thread = multiprocessing.Process(
            target=self._monitor_downloading, args=[duration, interval_in_seconds]
        )
        return thread

    def _monitor_downloading(self, duration, interval_in_seconds):
        initial_io = psutil.net_io_counters()

        prev_MB_received = 0

        for _ in range(duration // interval_in_seconds):

            # consider downloading finished if we have a significantly strong signal indicating loading is in progress
            if self.percent_progress_load.value > 2 or self.loading_failed.value:
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
            if self.loading_is_done.value or self.loading_failed.value:
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
            if self.loading_is_done.value or self.loading_failed.value:
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
        if (
            self.logging_enabled
            and self.prev_dl_value != self.percent_progress_download.value
        ):
            self.prev_dl_value = self.percent_progress_download.value

            self.logger(f"Percent downloaded: {self.percent_progress_download.value}%")

    def log_percent_done_loaded(self):
        if (
            self.logging_enabled
            and self.prev_load_value != self.percent_progress_load.value
        ):
            self.prev_load_value = self.percent_progress_load.value

            self.logger(
                f"Percent loaded into memory: {self.percent_progress_load.value}%"
            )
