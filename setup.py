#!/usr/bin/env python3
"""
Setup Script — AI Video Dubber
================================
Automated setup for all dependencies and models.

Usage:
    python setup.py                    # Full setup
    python setup.py --setup-wav2lip    # Only Wav2Lip
    python setup.py --check            # Verify installation
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, description=""):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  $ {cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ⚠️  Command exited with code {result.returncode}")
    return result.returncode == 0


def install_base_dependencies():
    """Install base Python packages."""
    print("\n📦 Installing base dependencies...")
    run_cmd(
        f"{sys.executable} -m pip install -q --upgrade pip",
        "Upgrading pip"
    )
    run_cmd(
        f"{sys.executable} -m pip install -q -r requirements.txt",
        "Installing requirements.txt"
    )


def install_torch_cuda():
    """Install PyTorch with CUDA support."""
    print("\n🔥 Installing PyTorch with CUDA...")
    run_cmd(
        f"{sys.executable} -m pip install -q torch torchvision torchaudio "
        f"--index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch (CUDA 11.8)"
    )


def _patch_wav2lip_numpy(wav2lip_dir):
    """Patch Wav2Lip source files for numpy >= 1.24 compatibility.

    np.float, np.int, np.complex were deprecated in numpy 1.20 and
    removed in numpy 1.24. Wav2Lip's code uses np.float in audio.py
    and potentially other files. This patches them to np.float64.
    """
    files_to_patch = [
        wav2lip_dir / "audio.py",
        wav2lip_dir / "models" / "wav2lip.py",
        wav2lip_dir / "models" / "wav2lip_gan.py",
    ]

    for fpath in files_to_patch:
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        patched = content.replace("np.float)", "np.float64)")
        patched = patched.replace("np.float,", "np.float64,")
        patched = patched.replace("np.int)", "np.int64)")
        patched = patched.replace("np.int,", "np.int64,")
        if patched != content:
            fpath.write_text(patched, encoding="utf-8")
            print(f"  ✅ Patched {fpath} for numpy compatibility")


def setup_wav2lip():
    """Clone and configure Wav2Lip."""
    print("\n👄 Setting up Wav2Lip...")

    wav2lip_dir = Path("Wav2Lip")

    if not wav2lip_dir.exists():
        run_cmd(
            "git clone https://github.com/Rudrabha/Wav2Lip.git",
            "Cloning Wav2Lip repository"
        )
    else:
        print("  Wav2Lip directory already exists")

    # Install Wav2Lip requirements
    wav2lip_reqs = wav2lip_dir / "requirements.txt"
    if wav2lip_reqs.exists():
        run_cmd(
            f"{sys.executable} -m pip install -q -r {wav2lip_reqs}",
            "Installing Wav2Lip dependencies"
        )

    # CRITICAL: Patch Wav2Lip for numpy >= 1.24 compatibility
    # np.float was removed in numpy 1.24; Wav2Lip's audio.py uses it
    _patch_wav2lip_numpy(wav2lip_dir)

    # Create checkpoints directory
    checkpoints = wav2lip_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    # Check for model files
    gan_model = checkpoints / "wav2lip_gan.pth"
    if not gan_model.exists():
        print("\n" + "="*60)
        print("  ⚠️  MANUAL STEP REQUIRED")
        print("  Download wav2lip_gan.pth from:")
        print("  https://iiitaphyd-my.sharepoint.com/:u:/g/personal/"
              "radrabha_m_research_iiit_ac_in/"
              "EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW")
        print(f"  Place it in: {gan_model.absolute()}")
        print("="*60)

    # Download face detection model
    face_det_dir = wav2lip_dir / "face_detection" / "detection" / "sfd"
    face_det_dir.mkdir(parents=True, exist_ok=True)
    s3fd_model = face_det_dir / "s3fd.pth"

    if not s3fd_model.exists():
        print("\n  Downloading face detection model (s3fd.pth)...")
        run_cmd(
            f"wget -q -O {s3fd_model} "
            "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
            "Downloading s3fd face detection model"
        )


def setup_indictrans2():
    """Install IndicTrans2 dependencies."""
    print("\n🌍 Setting up IndicTrans2...")

    run_cmd(
        f"{sys.executable} -m pip install -q IndicTransToolkit",
        "Installing IndicTransToolkit"
    )


def setup_gfpgan():
    """Install and configure GFPGAN."""
    print("\n✨ Setting up GFPGAN...")

    run_cmd(
        f"{sys.executable} -m pip install -q gfpgan",
        "Installing GFPGAN"
    )

    # GFPGAN auto-downloads model weights on first use
    print("  GFPGAN model will be downloaded on first run")


def check_installation():
    """Verify all components are properly installed."""
    print("\n🔍 Checking installation...\n")

    checks = {}

    # Python version
    py_ver = sys.version.split()[0]
    checks["Python"] = f"{py_ver} ✅" if sys.version_info >= (3, 10) else f"{py_ver} ⚠️ (3.10+ recommended)"

    # FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.split("\n")[0]
        checks["FFmpeg"] = f"{version[:40]} ✅"
    except FileNotFoundError:
        checks["FFmpeg"] = "NOT FOUND ❌"

    # PyTorch + CUDA
    try:
        import torch
        cuda = "CUDA " + torch.version.cuda if torch.cuda.is_available() else "CPU only"
        checks["PyTorch"] = f"{torch.__version__} ({cuda}) ✅"
    except ImportError:
        checks["PyTorch"] = "NOT INSTALLED ❌"

    # Whisper
    try:
        import whisper
        checks["Whisper"] = f"Available ✅"
    except ImportError:
        checks["Whisper"] = "NOT INSTALLED ❌"

    # Translation
    try:
        from deep_translator import GoogleTranslator
        checks["deep_translator"] = "Available ✅"
    except ImportError:
        checks["deep_translator"] = "NOT INSTALLED ❌"

    try:
        from IndicTransToolkit import IndicProcessor
        checks["IndicTrans2"] = "Available ✅"
    except ImportError:
        checks["IndicTrans2"] = "Not installed (optional) ⚠️"

    # TTS
    try:
        from TTS.api import TTS
        checks["Coqui TTS"] = "Available ✅"
    except ImportError:
        checks["Coqui TTS"] = "NOT INSTALLED ❌"

    # Edge TTS
    try:
        import edge_tts
        checks["Edge TTS"] = "Available ✅"
    except ImportError:
        checks["Edge TTS"] = "Not installed (optional) ⚠️"

    # GFPGAN
    try:
        from gfpgan import GFPGANer
        checks["GFPGAN"] = "Available ✅"
    except ImportError:
        checks["GFPGAN"] = "Not installed (optional) ⚠️"

    # Wav2Lip
    wav2lip_path = Path("Wav2Lip")
    if wav2lip_path.exists():
        model_exists = (wav2lip_path / "checkpoints" / "wav2lip_gan.pth").exists()
        if model_exists:
            checks["Wav2Lip"] = "Ready ✅"
        else:
            checks["Wav2Lip"] = "Repo exists but model missing ⚠️"
    else:
        checks["Wav2Lip"] = "Not cloned ❌"

    # Print results
    print("  Component Status:")
    print("  " + "-" * 50)
    for component, status in checks.items():
        print(f"  {component:20s} {status}")
    print()

    return checks


def main():
    parser = argparse.ArgumentParser(description="AI Video Dubber Setup")
    parser.add_argument("--check", action="store_true", help="Check installation only")
    parser.add_argument("--setup-wav2lip", action="store_true", help="Setup Wav2Lip only")
    parser.add_argument("--setup-indictrans2", action="store_true", help="Setup IndicTrans2 only")
    parser.add_argument("--no-wav2lip", action="store_true", help="Skip Wav2Lip setup")
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only PyTorch")
    args = parser.parse_args()

    if args.check:
        check_installation()
        return

    if args.setup_wav2lip:
        setup_wav2lip()
        return

    if args.setup_indictrans2:
        setup_indictrans2()
        return

    # Full setup
    print("🚀 AI Video Dubber — Full Setup")
    print("=" * 60)

    if not args.cpu:
        install_torch_cuda()

    install_base_dependencies()
    setup_indictrans2()
    setup_gfpgan()

    if not args.no_wav2lip:
        setup_wav2lip()

    print("\n" + "=" * 60)
    print("  Setup complete! Running verification...")
    print("=" * 60)

    check_installation()


if __name__ == "__main__":
    main()
