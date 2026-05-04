#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import traceback


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024


def print_section(title):
    print(f"\n[GPU-TEST] {title}")


def run_test(name, fn):
    print(f"[GPU-TEST] TEST {name}: start")
    try:
        result = fn()
        if result is not None:
            print(f"[GPU-TEST] TEST {name}: OK - {result}")
        else:
            print(f"[GPU-TEST] TEST {name}: OK")
        return True
    except Exception as exc:
        print(f"[GPU-TEST] TEST {name}: FAILED - {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def cuda_memory_line(torch, device):
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return (
            f"free={format_bytes(free_bytes)} total={format_bytes(total_bytes)} "
            f"allocated={format_bytes(allocated)} reserved={format_bytes(reserved)}"
        )
    except Exception as exc:
        return f"memory info unavailable: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Test GPU/CUDA from the same .venv used by main_custom.py")
    parser.add_argument(
        "--stress-mb",
        type=int,
        default=256,
        help="Temporary GPU allocation size in MB for a simple memory stress test. Use 0 to disable.",
    )
    args = parser.parse_args()

    print_section("Python Environment")
    print(f"[GPU-TEST] executable={sys.executable}")
    print(f"[GPU-TEST] version={sys.version.replace(os.linesep, ' ')}")
    print(f"[GPU-TEST] platform={platform.platform()}")
    print(f"[GPU-TEST] VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '')}")
    print(f"[GPU-TEST] cwd={os.getcwd()}")

    print_section("Selected Environment Variables")
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "COMFYUI_FORCE_CUDA_MALLOC",
        "COMFYUI_PYTORCH_INDEX_URL",
        "COMFYUI_TORCH_VERSION",
    ]
    print(json.dumps({key: os.environ.get(key, "") for key in env_keys}, indent=2))

    print_section("Torch Import")
    try:
        import torch
    except Exception as exc:
        print(f"[GPU-TEST] torch import FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2

    print(f"[GPU-TEST] torch.__version__={torch.__version__}")
    print(f"[GPU-TEST] torch.version.cuda={getattr(torch.version, 'cuda', None)}")
    print(f"[GPU-TEST] cuda.is_available={torch.cuda.is_available()}")
    print(f"[GPU-TEST] cuda.device_count={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("[GPU-TEST] RESULT: CUDA is not available to torch in this environment.")
        return 1

    device = torch.device("cuda:0")
    print_section("CUDA Device")
    print(f"[GPU-TEST] current_device={torch.cuda.current_device()}")
    print(f"[GPU-TEST] name={torch.cuda.get_device_name(device)}")
    print(f"[GPU-TEST] capability={torch.cuda.get_device_capability(device)}")
    print(f"[GPU-TEST] memory_before={cuda_memory_line(torch, device)}")

    failures = 0

    def sync():
        torch.cuda.synchronize(device)

    def test_basic_alloc():
        x = torch.ones((1024, 1024), device=device, dtype=torch.float32)
        y = x * 2.0
        sync()
        return f"sum={float(y.sum().detach().cpu())}"

    failures += 0 if run_test("float32 allocation/math", test_basic_alloc) else 1

    def test_matmul(dtype):
        def _inner():
            x = torch.randn((1024, 1024), device=device, dtype=dtype)
            y = torch.randn((1024, 1024), device=device, dtype=dtype)
            z = x @ y
            sync()
            return f"dtype={dtype} shape={tuple(z.shape)}"
        return _inner

    failures += 0 if run_test("float16 matmul", test_matmul(torch.float16)) else 1
    failures += 0 if run_test("bfloat16 matmul", test_matmul(torch.bfloat16)) else 1

    if hasattr(torch, "float8_e4m3fn"):
        def test_float8_cast():
            source = torch.randn((1024, 1024), device=device, dtype=torch.float16)
            fp8 = source.to(torch.float8_e4m3fn)
            restored = fp8.to(torch.bfloat16)
            sync()
            return f"fp8_dtype={fp8.dtype} restored_dtype={restored.dtype}"

        failures += 0 if run_test("float8_e4m3fn cast roundtrip", test_float8_cast) else 1
    else:
        print("[GPU-TEST] TEST float8_e4m3fn cast roundtrip: SKIPPED - dtype not exposed by torch")

    if args.stress_mb > 0:
        def test_stress_alloc():
            numel = args.stress_mb * 1024 * 1024 // 2
            x = torch.empty((numel,), device=device, dtype=torch.float16)
            x.fill_(1)
            sync()
            return f"allocated_test_tensor={args.stress_mb}MB dtype=float16"

        failures += 0 if run_test(f"temporary {args.stress_mb}MB GPU allocation", test_stress_alloc) else 1

    print(f"[GPU-TEST] memory_after={cuda_memory_line(torch, device)}")
    if failures:
        print(f"[GPU-TEST] RESULT: GPU is visible, but {failures} diagnostic test(s) failed.")
        return 1

    print("[GPU-TEST] RESULT: GPU basic CUDA/torch diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
