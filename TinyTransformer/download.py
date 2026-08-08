r"""Download GPT-2 model files (safetensors + vocab) via direct HTTP.

Bypasses huggingface_hub's Xet backend (which hf-mirror.com doesn't support)
by downloading directly from {endpoint}/{repo}/resolve/main/{file}.

Usage:
    py download.py                       # defaults: gpt2 -> ./weights
    py download.py --variant gpt2-xl     # gpt2 / gpt2-medium / gpt2-large / gpt2-xl
    py download.py --out D:\models\gpt2

Environment:
    HF_ENDPOINT   mirror URL (e.g. https://hf-mirror.com)
    HF_TOKEN      optional auth token
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO_ID = "gpt2"
ALLOWED_VARIANTS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

# Files to fetch from the HF repo.
NEEDED_FILES = [
    "model.safetensors",
    "merges.txt",
    "vocab.json",
    "config.json",
]
# main.cpp loads "encoder.json"; HF GPT-2 splits OpenAI's encoder.json into
# vocab.json + merges.txt. We alias merges.txt -> encoder.json after download.
LOADER_ALIASES = {"encoder.json": "merges.txt"}


def download_file(url: str, dst: Path, token: str | None) -> None:
    """Download a file with resume support via HTTP Range."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable — file already complete
            tmp.rename(dst)
            return
        raise

    total = resp.headers.get("Content-Length")
    total_int = int(total) + existing if total else None
    mode = "ab" if existing else "wb"

    print(f"  {dst.name}: ", end="", flush=True)
    downloaded = existing
    with open(tmp, mode) as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_int:
                pct = downloaded * 100 // total_int
                print(f"\r  {dst.name}: {downloaded >> 20} / {total_int >> 20} MiB ({pct}%)", end="", flush=True)
            else:
                print(f"\r  {dst.name}: {downloaded >> 20} MiB", end="", flush=True)
    print(" done")
    tmp.rename(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GPT-2 weights for TinyTransformer")
    parser.add_argument(
        "--variant",
        choices=ALLOWED_VARIANTS,
        default=REPO_ID,
        help="GPT-2 model size; tries <variant> then openai-community/<variant>",
    )
    parser.add_argument("--out", default=None, help="Output directory (default: <script_dir>/weights)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out).resolve() if args.out else script_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_candidates = [args.variant, f"openai-community/{args.variant}"]
    token = os.environ.get("HF_TOKEN")
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")

    # Probe which repo exists by HEAD-requesting config.json.
    repo_id = None
    for candidate in repo_candidates:
        probe_url = f"{endpoint}/{candidate}/resolve/main/config.json"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        req = urllib.request.Request(probe_url, headers=headers, method="HEAD")
        try:
            urllib.request.urlopen(req, timeout=15)
            repo_id = candidate
            break
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue
    if repo_id is None:
        sys.stderr.write(f"No reachable repo among: {repo_candidates}\n")
        return 1

    print(f"Repo     : {repo_id}")
    print(f"Endpoint : {endpoint}")
    print(f"Output   : {out_dir}")
    print(f"Token    : {'present' if token else '(anonymous)'}")
    print()

    for fname in NEEDED_FILES:
        url = f"{endpoint}/{repo_id}/resolve/main/{fname}"
        dst = out_dir / fname
        if dst.exists():
            print(f"  {fname}: already exists, skip")
            continue
        try:
            download_file(url, dst, token)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                sys.stderr.write(
                    f"\nAuthorization required for {fname}.\n"
                    "Get a read-only token from https://huggingface.co/settings/tokens\n"
                    "then re-run with: HF_TOKEN=hf_... py download.py\n"
                )
                return 1
            elif e.code == 404:
                sys.stderr.write(f"\nFile not found: {url}\n")
                return 1
            else:
                sys.stderr.write(f"\nHTTP {e.code} for {fname}: {e.reason}\n")
                return 1
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"\nDownload failed for {fname}: {exc}\n")
            return 1

    # Create aliases for files the C++ loader expects by a different name.
    for alias, src_name in LOADER_ALIASES.items():
        src = out_dir / src_name
        dst = out_dir / alias
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  {src_name} -> {alias} (alias for C++ loader)")

    # Verify all files the loader needs are present.
    loader_files = ["model.safetensors"] + list(LOADER_ALIASES.keys()) + ["vocab.json"]
    missing = [f for f in loader_files if not (out_dir / f).exists()]
    if missing:
        sys.stderr.write(f"Missing required files: {missing}\n")
        return 3

    print("\nDone. Files in place:")
    for name in NEEDED_FILES + list(LOADER_ALIASES.keys()):
        p = out_dir / name
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        tag = "OK" if p.exists() else "MISSING"
        print(f"  [{tag}] {name:<20s}  {size_kb:10.1f} KB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
