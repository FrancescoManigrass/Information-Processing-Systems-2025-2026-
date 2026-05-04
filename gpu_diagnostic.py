#!/usr/bin/env python3
import json
import os
import platform
import subprocess
import sys
import traceback


LOCAL_VENV_DIRNAME = ".venv"


def _repo_root():
    return os.path.dirname(os.path.realpath(__file__))


def _python_from_venv_dir(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _candidate_venv_dirs():
    explicit_venv = os.environ.get("GPU_DIAGNOSTIC_VENV", "").strip()
    if explicit_venv:
        yield explicit_venv

    # When launched by VS Code Code Runner, __file__ and cwd can differ.
    # Check both, then the CrownLabs workspace path used by main_custom.py.
    yield os.path.join(os.getcwd(), LOCAL_VENV_DIRNAME)
    yield os.path.join(_repo_root(), LOCAL_VENV_DIRNAME)
    yield "/vscode/workspace/.venv"


def _venv_python():
    seen = set()
    for venv_dir in _candidate_venv_dirs():
        if not venv_dir:
            continue

        venv_dir = os.path.realpath(venv_dir)
        if venv_dir in seen:
            continue
        seen.add(venv_dir)

        python_path = os.path.abspath(_python_from_venv_dir(venv_dir))
        if os.path.isfile(python_path):
            return python_path

    return None


def _can_import_torch_here():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _reexec_into_main_custom_venv():
    if os.environ.get("_GPU_DIAGNOSTIC_VENV_ACTIVE") == "1":
        return

    if _can_import_torch_here():
        return

    venv_python = _venv_python()
    if not venv_python:
        print("[GPU-DIAG] No usable venv python found.")
        print("[GPU-DIAG] Checked:")
        for candidate in _candidate_venv_dirs():
            print(f"[GPU-DIAG]   - {candidate}")
        print("[GPU-DIAG] Activate/create /vscode/workspace/.venv, then rerun.")
        return

    venv_python = os.path.abspath(venv_python)
    current_python = os.path.abspath(sys.executable)
    if current_python == venv_python:
        return

    print(f"[GPU-DIAG] Current Python cannot import torch: {current_python}")
    print(f"[GPU-DIAG] Re-launching with venv Python: {venv_python}")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = os.path.dirname(os.path.dirname(venv_python))
    env["_GPU_DIAGNOSTIC_VENV_ACTIVE"] = "1"
    env["PATH"] = os.path.dirname(venv_python) + os.pathsep + env.get("PATH", "")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(venv_python, [venv_python] + sys.argv, env)


def _format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024


def _run_command(command):
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except Exception as exc:
        return None, str(exc)


def _print_header():
    print("[GPU-DIAG] Script path:", os.path.realpath(__file__))
    print("[GPU-DIAG] Working directory:", os.getcwd())
    print("[GPU-DIAG] Python executable:", sys.executable)
    print("[GPU-DIAG] Python version:", sys.version.replace("\n", " "))
    print("[GPU-DIAG] Platform:", platform.platform())
    print("[GPU-DIAG] VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV", ""))
    print("[GPU-DIAG] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print("[GPU-DIAG] PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))

    code, output = _run_command(["nvidia-smi"])
    if code == 0:
        print("[GPU-DIAG] nvidia-smi: OK")
        print(output)
    else:
        print(f"[GPU-DIAG] nvidia-smi unavailable or failed: {output}")


def _test_torch_cuda():
    try:
        import torch
    except Exception:
        print("[GPU-DIAG] ERROR: cannot import torch")
        venv_python = _venv_python()
        if venv_python:
            print(f"[GPU-DIAG] A venv Python exists here: {venv_python}")
            print(f"[GPU-DIAG] Run this exact command:")
            print(f"[GPU-DIAG]   {venv_python} {os.path.realpath(__file__)}")
        else:
            print("[GPU-DIAG] No venv Python found. Expected one of:")
            for candidate in _candidate_venv_dirs():
                print(f"[GPU-DIAG]   - {_python_from_venv_dir(candidate)}")
        traceback.print_exc()
        return 1

    print("[GPU-DIAG] torch version:", torch.__version__)
    print("[GPU-DIAG] torch cuda build:", getattr(torch.version, "cuda", None))
    print("[GPU-DIAG] cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("[GPU-DIAG] RESULT: CUDA is not available to PyTorch in this environment.")
        return 2

    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        print("[GPU-DIAG] current device:", device_index)
        print("[GPU-DIAG] device name:", torch.cuda.get_device_name(device_index))
        print("[GPU-DIAG] compute capability:", f"{props.major}.{props.minor}")
        print("[GPU-DIAG] total memory:", _format_bytes(props.total_memory))
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        print("[GPU-DIAG] free memory:", _format_bytes(free_bytes))
        print("[GPU-DIAG] mem_get_info total:", _format_bytes(total_bytes))
    except Exception:
        print("[GPU-DIAG] ERROR: CUDA device query failed")
        traceback.print_exc()
        return 3

    tests = [
        ("float32 small matmul", torch.float32, 1024),
        ("float16 small matmul", torch.float16, 1024),
    ]

    if hasattr(torch, "bfloat16"):
        tests.append(("bfloat16 small matmul", torch.bfloat16, 1024))

    failures = 0
    for label, dtype, size in tests:
        try:
            torch.cuda.empty_cache()
            a = torch.randn((size, size), device="cuda", dtype=dtype)
            b = torch.randn((size, size), device="cuda", dtype=dtype)
            c = a @ b
            torch.cuda.synchronize()
            print(
                f"[GPU-DIAG] OK: {label}; result dtype={c.dtype}; "
                f"allocated={_format_bytes(torch.cuda.memory_allocated())}; "
                f"reserved={_format_bytes(torch.cuda.memory_reserved())}"
            )
            del a, b, c
        except Exception:
            failures += 1
            print(f"[GPU-DIAG] ERROR: {label} failed")
            traceback.print_exc()

    try:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        target_bytes = min(int(free_bytes * 0.25), 1024 * 1024 * 1024)
        num_float32 = max(target_bytes // 4, 1)
        x = torch.empty((num_float32,), device="cuda", dtype=torch.float32)
        x.fill_(1.0)
        torch.cuda.synchronize()
        print(f"[GPU-DIAG] OK: allocation test { _format_bytes(target_bytes) }")
        del x
    except Exception:
        failures += 1
        print("[GPU-DIAG] ERROR: allocation test failed")
        traceback.print_exc()

    fp8_report = {"available": False}
    for attr in ("float8_e4m3fn", "float8_e5m2"):
        if hasattr(torch, attr):
            fp8_report[attr] = True
            fp8_report["available"] = True
        else:
            fp8_report[attr] = False
    print("[GPU-DIAG] torch fp8 dtype availability:", json.dumps(fp8_report, sort_keys=True))

    if failures:
        print(f"[GPU-DIAG] RESULT: GPU detected, but {failures} CUDA test(s) failed.")
        return 4

    print("[GPU-DIAG] RESULT: GPU/CUDA basic tests passed.")
    return 0


def main():
    _reexec_into_main_custom_venv()
    _print_header()
    return _test_torch_cuda()


if __name__ == "__main__":
    raise SystemExit(main())
