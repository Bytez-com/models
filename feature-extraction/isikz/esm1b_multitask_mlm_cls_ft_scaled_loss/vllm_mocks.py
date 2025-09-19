from vllm.utils import MemorySnapshot
import vllm.utils


og_measure = MemorySnapshot.measure

# We've set a future proof insanely high number
# we do this to subvert the memory checks that are done by the program
# because we don't care about them
# NOTE careful of this on subsequent vLLM updates
EiB = 1024**6

free_memory = EiB


mocked_available_memory = free_memory


def mocked_measure(self: MemorySnapshot):
    # this is void
    og_measure(self)

    self.free_memory = mocked_available_memory


MemorySnapshot.measure = mocked_measure

vllm.utils.MemorySnapshot = MemorySnapshot

from vllm.v1.worker.gpu_worker import Worker
import vllm.v1.worker.gpu_worker


def mocked_determine_available_memory(self: Worker):
    return mocked_available_memory


vllm.v1.worker.gpu_worker.Worker.determine_available_memory = (
    mocked_determine_available_memory
)
