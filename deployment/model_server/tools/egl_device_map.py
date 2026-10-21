# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Resolve physical NVIDIA GPU indices to EGL device indices via PCI addresses.
"""Print EGL indices corresponding to positional nvidia-smi GPU indices."""

import argparse
import ctypes
import subprocess
from pathlib import Path


def _pci_address(value: str) -> tuple[int, int, int, int]:
    domain, bus, device_function = value.lower().split(":")
    device, function = device_function.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def resolve_egl_devices(gpu_ids: list[int]) -> list[int]:
    """Match NVIDIA EGL DRM nodes to nvidia-smi PCI IDs; fail on ambiguous mappings."""
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"], text=True
    )
    gpu_pci = {}
    for line in inventory.splitlines():
        index, address = line.split(",")
        gpu_pci[int(index)] = _pci_address(address.strip())

    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    query_address = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    string_address = egl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
    if not query_address or not string_address:
        raise RuntimeError("EGL device enumeration extensions are unavailable")
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(
        query_address
    )
    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(string_address)
    count = ctypes.c_int()
    if not query(0, None, ctypes.byref(count)) or count.value <= 0:
        raise RuntimeError("EGL reported no devices")
    devices = (ctypes.c_void_p * count.value)()
    if not query(len(devices), devices, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    pci_to_egl = {}
    for index, device in enumerate(devices[: count.value]):
        extensions = (query_string(device, 0x3055) or b"").split()  # EGL_EXTENSIONS
        # Mesa can enumerate the same DRM nodes; select NVIDIA's hardware devices.
        if b"EGL_NV_device_cuda" not in extensions:
            continue
        node = query_string(device, 0x3377) or query_string(device, 0x3233)  # DRM render node / device
        if not node:
            continue
        device_path = Path("/sys/class/drm") / Path(node.decode()).name / "device"
        if not device_path.exists():
            continue
        pci = _pci_address(device_path.resolve().name)
        if pci in pci_to_egl:
            raise RuntimeError(f"Multiple NVIDIA EGL devices match PCI address {pci}")
        pci_to_egl[pci] = index
    result = []
    for gpu in gpu_ids:
        if gpu not in gpu_pci or gpu_pci[gpu] not in pci_to_egl:
            raise RuntimeError(f"Cannot resolve physical GPU {gpu} to an NVIDIA EGL device")
        result.append(pci_to_egl[gpu_pci[gpu]])
    return result


def main() -> None:
    """Resolve physical indices, optionally validating an explicit mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_ids", nargs="+", type=int)
    parser.add_argument("--expected", nargs="+", type=int)
    args = parser.parse_args()
    resolved = resolve_egl_devices(args.gpu_ids)
    if args.expected is not None and resolved != args.expected:
        parser.error(f"Explicit EGL mapping {args.expected} disagrees with detected mapping {resolved}")
    print(" ".join(map(str, resolved)))


if __name__ == "__main__":
    main()
