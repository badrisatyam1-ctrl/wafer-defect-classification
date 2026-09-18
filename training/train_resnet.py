"""
Industry-style training entrypoint for full-wafer ResNet18 classification.

Key upgrades over a minimal training loop:
- Full-wafer inputs only; no patching or random crops.
- Lot/time holdout instead of wafer-level random split.
- Class-weighted CrossEntropy or Focal Loss, optionally paired with a
  WeightedRandomSampler for rare macro defect classes.
- Mixed precision, gradient clipping, checkpoint resume, early stopping,
  and validation artifacts for auditability.
- Checkpoints persist preprocessing metadata so inference is guaranteed to
  match the deterministic training preprocessor.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet18_classifier import (
    DEFECT_CLASSES_V2,
    FocalLoss,
    NUM_CLASSES,
    PreprocessingConfig,
    WaferPreprocessor,
    create_resnet18_classifier,
)
from utils.synthetic_generator import create_classification_dataset_v2
from utils.wafer_dataset import (
    WaferDatasetBundle,
    WaferMapDataset,
    compute_class_weights,
    compute_sample_weights,
    split_dataset_bundle,
)


@dataclass
class TrainConfig:
    """Config for a resumable, auditable training run."""

    dataset_npz: Optional[Path] = None
    resume_from: Optional[Path] = None
    split_mode: str = "lot"
    val_fraction: float = 0.2
    synthetic_samples: int = 10000
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    loss_name: str = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    pretrained: bool = True
    freeze_backbone: bool = False
    weighted_sampling: bool = True
    mixed_precision: bool = True
    grad_clip_norm: Optional[float] = 1.0
    patience: int = 5
    min_epochs_before_stop: int = 5
    num_workers: int = 0
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "checkpoints")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "resnet")
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _class_distribution(labels: np.ndarray) -> Dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=NUM_CLASSES)
    return {DEFECT_CLASSES_V2[idx]: int(count) for idx, count in enumerate(counts)}


def load_dataset_bundle(config: TrainConfig) -> WaferDatasetBundle:
    """
    Load a real dataset bundle from NPZ or fall back to the synthetic full-wafer set.

    Expected NPZ keys:
    - images: (N, H, W[, C])
    - labels: integer labels or class-name strings
    - lot_ids: required when split_mode='lot'
    - timestamps: required when split_mode='time'
    """

    if config.dataset_npz is None:
        images, labels, lot_ids = create_classification_dataset_v2(
            n_samples=config.synthetic_samples,
            size=(config.preprocessing.input_size, config.preprocessing.input_size),
        )
        unique_lots = list(dict.fromkeys(lot_ids.tolist()))
        lot_to_time = {lot_id: order for order, lot_id in enumerate(unique_lots)}
        timestamps = np.array([lot_to_time[lot_id] for lot_id in lot_ids], dtype=np.int64)
        bundle = WaferDatasetBundle(
            images=images,
            labels=labels,
            lot_ids=lot_ids,
            timestamps=timestamps,
        )
    else:
        with np.load(config.dataset_npz, allow_pickle=True) as payload:
            images = payload["images"]
            labels = payload["labels"]
            lot_ids = payload["lot_ids"] if "lot_ids" in payload.files else None
            timestamps = payload["timestamps"] if "timestamps" in payload.files else None

        if labels.dtype.kind in {"U", "S", "O"}:
            label_to_index = {name: idx for idx, name in enumerate(DEFECT_CLASSES_V2)}
            labels = np.array([label_to_index[str(label)] for label in labels], dtype=np.int64)
        else:
            labels = labels.astype(np.int64)

        bundle = WaferDatasetBundle(
            images=images,
            labels=labels,
            lot_ids=lot_ids,
            timestamps=timestamps,
        )

    if len(bundle.labels) == 0:
        raise ValueError("Dataset is empty.")
    if bundle.labels.min() < 0 or bundle.labels.max() >= NUM_CLASSES:
        raise ValueError("Labels must map to the configured macro defect classes.")

    return bundle


# ─────────────────────────────────────────────────────────────────────
# Domain-expert class weights (user-specified)
# normal=1.0, center=3.5, edge_ring=4.0, edge_loss=4.0,
# scratch=3.0, ring=3.5, cluster=3.0, full_fail=5.0
# ─────────────────────────────────────────────────────────────────────
CUSTOM_CLASS_WEIGHTS = torch.tensor(
    [1.0, 3.5, 4.0, 4.0, 3.0, 3.5, 3.0, 5.0]
)


def build_loss(config: TrainConfig, class_weights: torch.Tensor) -> nn.Module:
    # Avoid double-compensating for class imbalance
    weights_to_use = None if config.weighted_sampling else class_weights
    if config.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights_to_use, label_smoothing=config.label_smoothing)
    if config.loss_name == "focal":
        return FocalLoss(gamma=config.focal_gamma, alpha=weights_to_use, label_smoothing=config.label_smoothing)
    raise ValueError("loss_name must be 'focal' or 'cross_entropy'.")


def make_train_loader(
    dataset: WaferMapDataset,
    labels: np.ndarray,
    config: TrainConfig,
) -> DataLoader:
    if config.weighted_sampling:
        sample_weights = compute_sample_weights(labels, num_classes=NUM_CLASSES)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.device == "cuda",
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp_enabled: bool = False,
    grad_clip_norm: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(mode=training)

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if amp_enabled
        else nullcontext()
    )

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context:
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        
        # --- PER-STEP UPDATES (Real-time telemetry) ---
        if training:
            stats = {
                "epoch": kwargs.get("epoch_idx", 0),
                "step": len(all_predictions),
                "total_steps": len(loader.dataset),
                "loss": float(loss.item()),
                "timestamp": time.time()
            }
            with open(PROJECT_ROOT / "logs" / "live_train_stats.json", "w") as f:
                json.dump(stats, f)

        running_loss += float(loss.item()) * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())

    labels_np = np.asarray(all_labels, dtype=np.int64)
    predictions_np = np.asarray(all_predictions, dtype=np.int64)

    return {
        "loss": running_loss / max(1, len(labels_np)),
        "accuracy": accuracy_score(labels_np, predictions_np),
        "balanced_accuracy": balanced_accuracy_score(labels_np, predictions_np),
        "macro_recall": recall_score(labels_np, predictions_np, average="macro", zero_division=0),
        "macro_f1": f1_score(labels_np, predictions_np, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels_np, predictions_np, average="weighted", zero_division=0),
        "labels": labels_np,
        "predictions": predictions_np,
    }


def serialize_config(config: TrainConfig) -> Dict[str, object]:
    return {
        "dataset_npz": None if config.dataset_npz is None else str(config.dataset_npz),
        "resume_from": None if config.resume_from is None else str(config.resume_from),
        "split_mode": config.split_mode,
        "val_fraction": config.val_fraction,
        "synthetic_samples": config.synthetic_samples,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "loss_name": config.loss_name,
        "focal_gamma": config.focal_gamma,
        "label_smoothing": config.label_smoothing,
        "warmup_epochs": config.warmup_epochs,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "weighted_sampling": config.weighted_sampling,
        "mixed_precision": config.mixed_precision,
        "grad_clip_norm": config.grad_clip_norm,
        "patience": config.patience,
        "min_epochs_before_stop": config.min_epochs_before_stop,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "preprocessing": config.preprocessing.to_dict(),
        "checkpoint_dir": str(config.checkpoint_dir),
        "log_dir": str(config.log_dir),
        "device": config.device,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainConfig,
    epoch: int,
    best_macro_f1: float,
    class_weights: torch.Tensor,
    val_metrics: Dict[str, object],
) -> None:
    checkpoint = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "resnet18",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "best_macro_f1": best_macro_f1,
        "best_val_metrics": {
            key: value
            for key, value in val_metrics.items()
            if key not in {"labels", "predictions"}
        },
        "class_names": DEFECT_CLASSES_V2,
        "class_weights": class_weights.detach().cpu().tolist(),
        "preprocessing": config.preprocessing.to_dict(),
        "config": serialize_config(config),
        "split_mode": config.split_mode,
        "loss_name": config.loss_name,
        "focal_gamma": config.focal_gamma,
        "temperature": 1.0,
    }
    torch.save(checkpoint, checkpoint_path)


def load_resume_checkpoint(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
) -> Dict[str, object]:
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler.is_enabled() and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def save_validation_artifacts(
    log_dir: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    best_macro_f1: float,
    split_mode: str,
) -> None:
    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=DEFECT_CLASSES_V2,
        zero_division=0,
        output_dict=True,
    )
    report_path = log_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    matrix = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))
    matrix_path = log_dir / "confusion_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + DEFECT_CLASSES_V2)
        for class_name, row in zip(DEFECT_CLASSES_V2, matrix.tolist()):
            writer.writerow([class_name] + row)

    summary_path = log_dir / "run_summary.json"
    summary = {
        "best_macro_f1": best_macro_f1,
        "split_mode": split_mode,
        "validation_distribution": _class_distribution(labels),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def make_json_serializable(history: Dict[str, list]) -> Dict[str, list]:
    serializable = {}
    for split_name, metrics_list in history.items():
        serializable[split_name] = []
        for metrics in metrics_list:
            row = {}
            for key, value in metrics.items():
                if key in {"labels", "predictions"}:
                    continue
                if isinstance(value, np.generic):
                    row[key] = value.item()
                else:
                    row[key] = value
            serializable[split_name].append(row)
    return serializable


def main(config: TrainConfig):
    set_seed(config.seed)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RESNET18 FULL-WAFER TRAINING")
    print("=" * 72)
    print(f"device: {config.device}")
    print(f"split_mode: {config.split_mode}")
    print(f"loss: {config.loss_name}")
    print(f"weighted_sampling: {config.weighted_sampling}")
    print(f"mixed_precision: {config.mixed_precision and config.device == 'cuda'}")
    print(f"input_size: {config.preprocessing.input_size}")

    bundle = load_dataset_bundle(config)
    splits = split_dataset_bundle(
        bundle,
        split_mode=config.split_mode,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_bundle = splits["train"]
    val_bundle = splits["val"]

    print(f"train_samples: {len(train_bundle.labels)}")
    print(f"val_samples: {len(val_bundle.labels)}")
    print(f"train_distribution: {_class_distribution(train_bundle.labels)}")
    print(f"val_distribution: {_class_distribution(val_bundle.labels)}")
    if train_bundle.lot_ids is not None:
        print(f"train_lots: {len(np.unique(train_bundle.lot_ids))}")
    if val_bundle.lot_ids is not None:
        print(f"val_lots: {len(np.unique(val_bundle.lot_ids))}")

    preprocessor = WaferPreprocessor(config.preprocessing)
    train_dataset = WaferMapDataset(
        train_bundle.images,
        train_bundle.labels,
        preprocessor=preprocessor,
        training=True,
    )
    val_dataset = WaferMapDataset(
        val_bundle.images,
        val_bundle.labels,
        preprocessor=preprocessor,
        training=False,
    )

    train_loader = make_train_loader(train_dataset, train_bundle.labels, config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    model = create_resnet18_classifier(
        num_classes=NUM_CLASSES,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs - config.warmup_epochs),
        eta_min=1e-6,
    )
    if config.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=config.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_epochs]
        )
    else:
        scheduler = main_scheduler

    amp_enabled = config.mixed_precision and config.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Compute dynamic inverse-frequency class weights directly from training pool
    class_weights = compute_class_weights(train_bundle.labels, num_classes=NUM_CLASSES)
    class_weights = class_weights.to(config.device)
    
    criterion = build_loss(config, class_weights)
    print("class_weights:", np.round(class_weights.detach().cpu().numpy(), 3).tolist())

    checkpoint_path = config.checkpoint_dir / "resnet18_best.pt"
    last_checkpoint_path = config.checkpoint_dir / "resnet18_last.pt"
    history = {"train": [], "val": []}
    best_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    start_epoch = 1

    if config.resume_from is not None:
        resume_checkpoint = load_resume_checkpoint(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=config.device,
        )
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_macro_f1 = float(resume_checkpoint.get("best_macro_f1", float("-inf")))
        print(f"resumed_from: {config.resume_from}")
        print(f"resume_epoch: {start_epoch}")

    for epoch in range(start_epoch, config.epochs + 1):
        start_time = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            config.device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=config.grad_clip_norm,
            epoch_idx=epoch,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            config.device,
            optimizer=None,
            scaler=None,
            amp_enabled=False,
            grad_clip_norm=None,
        )
        scheduler.step()

        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        val_metrics["lr"] = optimizer.param_groups[0]["lr"]
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_recall={val_metrics['macro_recall']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{time.time() - start_time:.1f}s"
        )

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            best_macro_f1=max(best_macro_f1, val_metrics["macro_f1"]),
            class_weights=class_weights,
            val_metrics=val_metrics,
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                class_weights=class_weights,
                val_metrics=val_metrics,
            )
            print(f"  saved best checkpoint -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if (
            epoch >= config.min_epochs_before_stop
            and epochs_without_improvement >= config.patience
        ):
            print(f"early_stopping: no macro-F1 improvement for {config.patience} epochs")
            break

    best_checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_val = run_epoch(
        model,
        val_loader,
        criterion,
        config.device,
        optimizer=None,
        scaler=None,
        amp_enabled=False,
        grad_clip_norm=None,
    )

    print("=" * 72)
    print("FINAL VALIDATION REPORT")
    print("=" * 72)
    print(
        classification_report(
            final_val["labels"],
            final_val["predictions"],
            labels=list(range(NUM_CLASSES)),
            target_names=DEFECT_CLASSES_V2,
            zero_division=0,
        )
    )

    history_path = config.log_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(make_json_serializable(history), handle, indent=2)

    save_validation_artifacts(
        log_dir=config.log_dir,
        labels=final_val["labels"],
        predictions=final_val["predictions"],
        best_macro_f1=best_macro_f1,
        split_mode=config.split_mode,
    )

    print(f"history_path: {history_path}")
    print(f"best_checkpoint_path: {checkpoint_path}")
    print(f"last_checkpoint_path: {last_checkpoint_path}")
    print(f"best_macro_f1: {best_macro_f1:.4f}")

    return model, history


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train an industry-style ResNet18 wafer classifier.")
    parser.add_argument("--dataset-npz", type=Path, default=None, help="Optional NPZ dataset bundle.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a previous checkpoint.")
    parser.add_argument("--split-mode", choices=["lot", "time"], default="lot")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--synthetic-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["focal", "cross_entropy"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs-before-stop", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--disable-weighted-sampling", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        dataset_npz=args.dataset_npz,
        resume_from=args.resume_from,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        synthetic_samples=args.synthetic_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        weighted_sampling=not args.disable_weighted_sampling,
        mixed_precision=not args.disable_amp,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        min_epochs_before_stop=args.min_epochs_before_stop,
        num_workers=args.num_workers,
        seed=args.seed,
        preprocessing=PreprocessingConfig(input_size=args.input_size),
    )


if __name__ == "__main__":
    main(parse_args())
