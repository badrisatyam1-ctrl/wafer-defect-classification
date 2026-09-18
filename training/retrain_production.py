import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 80)
    print("SIH WAFER DEFECT CLASSIFICATION - PRODUCTION RETRAINING")
    print("=" * 80)
    
    npz_path = PROJECT_ROOT / "wafer_dataset_10k.npz"
    if not npz_path.exists():
        npz_path = PROJECT_ROOT / "dataset" / "wafer_dataset_10k.npz"
    use_npz = npz_path.exists()
    
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train_resnet.py"),
        "--epochs", "30",
        "--batch-size", "32",
        "--synthetic-samples", "10000",
        "--loss", "focal",
        "--label-smoothing", "0.1",
        "--warmup-epochs", "3",
        "--patience", "8"
    ]
    
    if use_npz:
        print(f"[OK] Found real dataset NPZ at {npz_path}. Using it for training.")
        cmd.extend(["--dataset-npz", str(npz_path)])
    else:
        print("[WARN] No real dataset NPZ found. Generating 10,000 synthetic samples...")
        
    print(f"\n[TRAIN] Running training with command:\n{' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Training failed with error code {e.returncode}")
        return
        
    best_checkpoint = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_best.pt"
    deploy_path = PROJECT_ROOT / "models" / "best.pt"
    
    if best_checkpoint.exists():
        print(f"\n[DONE] Training completed successfully!")
        print(f"[COPY] Copying best checkpoint to {deploy_path}")
        shutil.copy2(best_checkpoint, deploy_path)
        print("[READY] Model deployed for production inference!")
    else:
        print(f"\n[WARN] Training finished but couldn't find checkpoint at {best_checkpoint}")

if __name__ == "__main__":
    main()
