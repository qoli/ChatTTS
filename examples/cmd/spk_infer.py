#!/usr/bin/env python3
"""Generate audio with ChatTTS by loading a speaker embedding from a .pt file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

import ChatTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ChatTTS inference with a custom speaker embedding."
    )
    parser.add_argument(
        "--spk-path",
        default="pt/seed_2279_restored_emb.pt",
        help="Path to the torch .pt file that stores the speaker embedding.",
    )
    parser.add_argument(
        "--text",
        default="Hello from ChatTTS with a restored timbre.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="spk_control.wav",
        help="Destination wav filename for the generated audio.",
    )
    parser.add_argument(
        "--source",
        default="local",
        choices=["local", "huggingface", "custom"],
        help="Where ChatTTS should load weights from.",
    )
    parser.add_argument(
        "--custom-path",
        default="",
        help="When --source=custom, point to the directory containing asset/.",
    )
    return parser.parse_args()


def main() -> None:
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    args = parse_args()
    pt_path = Path(args.spk_path).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Speaker embedding file not found: {pt_path}")

    print(f"[INFO] Loading speaker embedding from {pt_path}")
    spk = torch.load(pt_path, map_location="cpu")

    print("[INFO] Initializing ChatTTS")
    chat = ChatTTS.Chat()

    if args.source == "custom" and args.custom_path:
        loaded = chat.load(source="custom", custom_path=args.custom_path)
    else:
        loaded = chat.load(source=args.source)

    if not loaded:
        raise RuntimeError("Failed to load ChatTTS weights. Check --source/--custom-path.")

    print("[INFO] Running inference with custom speaker embedding")
    params = ChatTTS.Chat.InferCodeParams(spk_emb=spk)
    wavs = chat.infer([args.text], params_infer_code=params)

    audio = torch.from_numpy(wavs[0]).unsqueeze(0)
    torchaudio.save(args.output, audio, 24000)
    print(f"[INFO] Saved synthesized audio to {args.output}")


if __name__ == "__main__":
    main()
