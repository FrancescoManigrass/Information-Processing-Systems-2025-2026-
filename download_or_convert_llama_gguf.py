#!/usr/bin/env python3
"""
Download a Llama-family model and make it usable by comfyui-sg-llama-cpp.

Modes:
  1) direct   -> download a ready-made GGUF file from Hugging Face
  2) convert  -> download a Hugging Face model repo, convert to GGUF with llama.cpp,
                 then optionally quantize it

Examples:
  # Easiest path: download a ready GGUF directly into your ComfyUI text_encoders folder
  python download_or_convert_llama_gguf.py \
    --mode direct \
    --direct-repo bartowski/Llama-3.2-3B-Instruct-GGUF \
    --direct-file Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --out "D:/ComfyUI/models/text_encoders"

  # Full path: download HF repo, convert to F16 GGUF, quantize to Q4_K_M, and place result in text_encoders
  python download_or_convert_llama_gguf.py \
    --mode convert \
    --hf-repo meta-llama/Llama-3.2-3B-Instruct \
    --quant Q4_K_M \
    --workdir "D:/llm_build" \
    --out "D:/ComfyUI/models/text_encoders"
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# --------------------------- helpers ---------------------------

def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_python_package(pkg_spec: str, import_name: str | None = None) -> None:
    import_name = import_name or pkg_spec.split("[")[0].split("=")[0].replace("-", "_")
    try:
        __import__(import_name)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-U", pkg_spec])


def find_executable(candidates: list[Path]) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Non trovo il binario di quantizzazione. Ho cercato in:\n  - "
        + "\n  - ".join(str(c) for c in candidates)
    )


def clone_or_update_llama_cpp(repo_dir: Path) -> None:
    if repo_dir.exists():
        if (repo_dir / ".git").exists():
            run(["git", "pull", "--ff-only"], cwd=repo_dir)
        else:
            raise RuntimeError(f"{repo_dir} esiste ma non è un repo git")
    else:
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(repo_dir)])


def build_llama_cpp(repo_dir: Path) -> Path:
    build_dir = repo_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    run(["cmake", "-S", str(repo_dir), "-B", str(build_dir)])
    run(["cmake", "--build", str(build_dir), "--config", "Release"])

    is_windows = platform.system().lower().startswith("win")
    candidates = [
        build_dir / "bin" / ("llama-quantize.exe" if is_windows else "llama-quantize"),
        build_dir / "bin" / "Release" / ("llama-quantize.exe" if is_windows else "llama-quantize"),
        repo_dir / ("llama-quantize.exe" if is_windows else "llama-quantize"),
        repo_dir / ("quantize.exe" if is_windows else "quantize"),
    ]
    return find_executable(candidates)


def download_direct_gguf(repo_id: str, filename: str, out_dir: Path, hf_token: str | None) -> Path:
    ensure_python_package("huggingface_hub[cli]", "huggingface_hub")
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=hf_token,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def snapshot_model_repo(repo_id: str, local_dir: Path, hf_token: str | None) -> Path:
    ensure_python_package("huggingface_hub[cli]", "huggingface_hub")
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        token=hf_token,
        # skip giant/special files that are not needed for GGUF conversion
        ignore_patterns=[
            "*.gguf",
            "*.onnx",
            "*.msgpack",
            "*.h5",
            "*.ot",
            "*.tflite",
            "*.tar.gz",
        ],
        local_dir_use_symlinks=False,
    )
    return local_dir


def convert_hf_to_f16_gguf(repo_dir: Path, model_dir: Path, out_dir: Path) -> Path:
    ensure_python_package("numpy", "numpy")
    ensure_python_package("sentencepiece", "sentencepiece")
    ensure_python_package("protobuf", "google.protobuf")
    ensure_python_package("transformers", "transformers")
    ensure_python_package("torch", "torch")
    ensure_python_package("safetensors", "safetensors")

    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        run([sys.executable, "-m", "pip", "install", "-U", "-r", str(req_file)])

    converter = repo_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Script non trovato: {converter}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_dir.name}-f16.gguf"

    cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        "f16",
    ]
    run(cmd, cwd=repo_dir)
    return out_file


def quantize_gguf(quantize_bin: Path, input_gguf: Path, quant_type: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_gguf.stem.replace("-f16", "")
    out_file = out_dir / f"{stem}-{quant_type}.gguf"
    run([str(quantize_bin), str(input_gguf), str(out_file), quant_type])
    return out_file


def copy_to_output(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    if src.resolve() != dst.resolve():
        print(f"\nCopio: {src} -> {dst}")
        shutil.copy2(src, dst)
    return dst


# --------------------------- main ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scarica o converte un modello Llama in GGUF per ComfyUI llama.cpp")
    p.add_argument("--mode", choices=["direct", "convert"], required=True,
                   help="direct = scarica un GGUF già pronto, convert = scarica repo HF e converte")
    p.add_argument("--out", required=True,
                   help="Cartella finale dove vuoi il .gguf (es. la tua text_encoders di ComfyUI)")
    p.add_argument("--workdir", default="./llm_work",
                   help="Cartella di lavoro per download, clone e build")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                   help="Token Hugging Face. In alternativa imposta HF_TOKEN come variabile ambiente")

    # direct mode
    p.add_argument("--direct-repo", default="bartowski/Llama-3.2-3B-Instruct-GGUF",
                   help="Repo HF da cui scaricare un GGUF pronto")
    p.add_argument("--direct-file", default="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                   help="File GGUF da scaricare in modalità direct")

    # convert mode
    p.add_argument("--hf-repo", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="Repo HF da convertire in GGUF in modalità convert")
    p.add_argument("--quant", default="Q4_K_M",
                   help="Quant finale, es. Q4_K_M, Q5_K_M, Q8_0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    hf_token = args.hf_token or None

    try:
        if args.mode == "direct":
            gguf = download_direct_gguf(args.direct_repo, args.direct_file, out_dir, hf_token)
            print(f"\nFatto. GGUF pronto in:\n{gguf}")
            print("Riavvia ComfyUI se il modello non compare subito nel menu.")
            return 0

        # convert mode
        model_dir = workdir / "hf_model"
        llama_cpp_dir = workdir / "llama.cpp"
        converted_dir = workdir / "converted"
        quantized_dir = workdir / "quantized"

        print(f"\nScarico il repo HF: {args.hf_repo}")
        snapshot_model_repo(args.hf_repo, model_dir, hf_token)

        print("\nPreparo llama.cpp")
        clone_or_update_llama_cpp(llama_cpp_dir)
        quantize_bin = build_llama_cpp(llama_cpp_dir)

        print("\nConverto in F16 GGUF")
        f16_gguf = convert_hf_to_f16_gguf(llama_cpp_dir, model_dir, converted_dir)

        print(f"\nQuantizzo in {args.quant}")
        quant_gguf = quantize_gguf(quantize_bin, f16_gguf, args.quant, quantized_dir)

        final_path = copy_to_output(quant_gguf, out_dir)
        print(f"\nFatto. GGUF finale in:\n{final_path}")
        print("Riavvia ComfyUI se il modello non compare subito nel menu.")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\nErrore durante l'esecuzione del comando: {e}", file=sys.stderr)
        return e.returncode or 1
    except Exception as e:
        print(f"\nErrore: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
