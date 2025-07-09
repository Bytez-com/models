import resource
from dataclasses import dataclass


@dataclass
class SystemRamTracker:
    baseline_utilization_GB: float = 0

    def get_system_ram_usage_GB(self):
        peak_memory_usage_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2
        )
        return peak_memory_usage_GB

    def set_baseline_utilization_GB(self):
        baseline_utilization_GB = self.get_system_ram_usage_GB()

        self.baseline_utilization_GB = baseline_utilization_GB

    def get_ram_stats(self):
        peak_system_ram_usage_GB = self.get_system_ram_usage_GB()

        peak_model_ram_usage_GB = (
            peak_system_ram_usage_GB - self.baseline_utilization_GB
        )

        return {
            "peak_system_ram_usage_GB": peak_system_ram_usage_GB,
            "peak_model_ram_usage_GB": peak_model_ram_usage_GB,
        }


SYSTEM_RAM_TRACKER = SystemRamTracker()
