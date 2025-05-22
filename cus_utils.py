from typing import Dict, Optional, Tuple, Union
import sys
import torch
import logging
from enum import Enum


class MemoryUnit(Enum):
    BYTES = 1
    KB = 1024
    MB = 1024**2
    GB = 1024**3
    TB = 1024**4


def convert_memory(bytes_value: int, unit: MemoryUnit = MemoryUnit.GB) -> float:
    """Convert memory from bytes to specified unit.

    Args:
        bytes_value: Memory value in bytes
        unit: Target memory unit from MemoryUnit enum

    Returns:
        float: Converted memory value
    """
    return bytes_value / unit.value


def get_gpu_properties() -> Optional[Dict[str, Union[str, float, Tuple[int, int]]]]:
    """Get GPU device properties if CUDA is available.

    Returns:
        Dict containing GPU properties or None if CUDA is not available
    """
    try:
        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "name": props.name,
            "total_memory": props.total_memory,
            "cuda_capability": (props.major, props.minor),
            "device_index": device,
        }
    except Exception as e:
        logging.error(f"Error getting GPU properties: {str(e)}")
        return None


def get_system_info() -> Dict[str, str]:
    """Get system version information.

    Returns:
        Dict containing Python and PyTorch version info
    """
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    }


def check_gpu_memory(
    unit: MemoryUnit = MemoryUnit.GB, logger: Optional[logging.Logger] = None
) -> None:
    """
    Check and display GPU memory usage and system information.

    Args:
        unit: Memory unit to display results in (default: GB)
        logger: Optional logger instance for logging output
    """

    def log_or_print(msg: str) -> None:
        if logger:
            logger.info(msg)
        else:
            print(msg)

    try:
        # Get system information
        sys_info = get_system_info()
        log_or_print("\n=== System Information ===")
        for key, value in sys_info.items():
            log_or_print(f"{key.replace('_', ' ').title()}: {value}")

        # Get GPU information
        gpu_props = get_gpu_properties()
        if not gpu_props:
            log_or_print("\nCUDA is not available. Running on CPU.")
            return

        log_or_print(f"\n=== GPU Information ({gpu_props['name']}) ===")
        log_or_print(f"CUDA Device Index: {gpu_props['device_index']}")
        log_or_print(
            f"CUDA Capability: {gpu_props['cuda_capability'][0]}.{gpu_props['cuda_capability'][1]}"
        )

        # Get memory information
        device = gpu_props["device_index"]
        total = convert_memory(gpu_props["total_memory"], unit)
        allocated = convert_memory(torch.cuda.memory_allocated(device), unit)
        cached = convert_memory(torch.cuda.memory_reserved(device), unit)
        free = total - allocated

        log_or_print(f"\n=== Memory Usage ({unit.name}) ===")
        log_or_print(f"Total Memory    : {total:.2f}")
        log_or_print(f"Allocated Memory: {allocated:.2f}")
        log_or_print(f"Cached Memory   : {cached:.2f}")
        log_or_print(f"Free Memory     : {free:.2f}")

    except Exception as e:
        error_msg = f"Error checking GPU memory: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
