import time
import torch
from loguru import logger

from magic_pdf.libs.clean_memory import clean_memory


def get_vram(
    device: torch.device,
):
    if torch.cuda.is_available() and device != 'cpu':
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return total_memory
    elif str(device).startswith("npu"):
        import torch_npu
        if torch_npu.npu.is_available():
            total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)
            return total_memory
    else:
        return None


def clean_vram(
    device: torch.device,
    vram_threshold: int = 8,
):
    total_memory = get_vram(device)
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        logger.info(f"gc time: {gc_time}")
