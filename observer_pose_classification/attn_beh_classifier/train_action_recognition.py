import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from models.action_recognition import ActionRecognitionModel
from datasets.action_recognition_dataset import ActionRecognitionDataset
from utils.data_utils import load_and_split_segments
from utils.training_utils import (
    ProgressiveLRScheduler,
    train_epoch,
    val_epoch,
    test_epoch,
)
from utils.output_utils import (
    setup_logging,
    plot_training_curves,
    save_best_model,
    save_checkpoint,
    save_validation_outputs,
)


def parse_args():
    """Parse command-line arguments for training configuration.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train action recognition model with progressive learning rates"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to directory with preloaded HDF5 files",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        help="Path to annotation file (optional if labels are in metadata)",
    )
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="Use random initialization instead of pretrained weights",
    )

    # Out-of-domain testing
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--test_annotation_file", type=str, default=None)
    parser.add_argument("--test_every_n_epochs", type=int, default=10)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Progressive learning rate parameters
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_start_lr", type=float, default=1e-6)
    parser.add_argument("--encoder_target_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exponential"],
    )

    # Data parameters
    parser.add_argument("--clip_length", type=int, default=20)
    parser.add_argument(
        "--temporal_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--balance_classes", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")

    # Video-based validation and K-fold
    parser.add_argument(
        "--split_by_video",
        action="store_true",
        help="Split by video/session instead of random segments",
    )
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=3)

    # Save validation outputs
    parser.add_argument("--save_val_outputs", action="store_true")
    parser.add_argument("--save_outputs_every_n_epochs", type=int, default=10)

    # Model parameters
    parser.add_argument("--use_attentive_classifier", action="store_true")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=224)

    # System parameters
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/action_recognition")
    parser.add_argument("--resume_from", type=str, default=None)

    # Performance optimization
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")

    return parser.parse_args()


def main():
    """Main training entry point.

    Loads data, constructs the model, runs the training loop with progressive
    learning rate scheduling, and saves checkpoints and training curves.
    """
    args = parse_args()

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.data.dataloader"
    )
    torch.set_num_threads(min(8, torch.get_num_threads()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Arguments: {args}")

    args.num_classes = 7

    logger.info("PROGRESSIVE TRAINING STRATEGY:")
    logger.info(f"  Classifier LR: {args.classifier_lr:.2e}")
    logger.info(f"  Encoder start LR: {args.encoder_start_lr:.2e}")
    logger.info(f"  Encoder target LR: {args.encoder_target_lr:.2e}")
    logger.info(f"  Warmup epochs: {args.warmup_epochs}")
    logger.info(f"  LR schedule: {args.lr_schedule}")

    if args.test_path:
        logger.info(
            f"OUT-OF-DOMAIN TESTING: {args.test_path}, every {args.test_every_n_epochs} epochs"
        )

    if args.fold_index is not None:
        if args.fold_index < 0 or args.fold_index >= args.n_folds:
            raise ValueError(
                f"fold_index ({args.fold_index}) must be between 0 and {args.n_folds - 1}"
            )
        logger.info(f"K-FOLD: fold {args.fold_index}/{args.n_folds - 1}")

    train_segments, val_segments, _ = load_and_split_segments(
        cache_dir=args.cache_dir,
        annotation_file=args.annotation_file,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        subsample=args.subsample,
        logger=logger,
        split_by_video=args.split_by_video,
        fold_index=args.fold_index,
        n_folds=args.n_folds,
    )

    train_dataset = ActionRecognitionDataset(
        segments=train_segments,
        cache_dir=args.cache_dir,
        clip_length=args.clip_length,
        temporal_sampling=args.temporal_sampling,
        split="train",
        use_augmentation=args.use_augmentation,
        balance_classes=args.balance_classes,
        logger=logger,
        target_size=(args.target_size, args.target_size),
    )

    val_dataset = ActionRecognitionDataset(
        segments=val_segments,
        cache_dir=args.cache_dir,
        clip_length=args.clip_length,
        temporal_sampling=args.temporal_sampling,
        split="val",
        use_augmentation=False,
        balance_classes=False,
        logger=logger,
        target_size=(args.target_size, args.target_size),
    )

    test_dataset = None
    test_loader = None
    test_segments = []
    if args.test_path:
        logger.info("Loading out-of-domain test dataset...")
        test_segments, _, _ = load_and_split_segments(
            cache_dir=args.test_path,
            annotation_file=args.test_annotation_file,
            val_ratio=None,
            random_state=args.random_state,
            subsample=1,
            logger=logger,
            split_by_video=False,
            fold_index=None,
            n_folds=3,
        )
        test_dataset = ActionRecognitionDataset(
            segments=test_segments,
            cache_dir=args.test_path,
            clip_length=args.clip_length,
            temporal_sampling=args.temporal_sampling,
            split="test",
            use_augmentation=False,
            balance_classes=False,
            logger=logger,
            target_size=(args.target_size, args.target_size),
        )

    train_sampler = train_dataset.get_sampler() if args.balance_classes else None
    optimal_workers = min(args.num_workers, torch.get_num_threads() // 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and optimal_workers > 0,
        prefetch_factor=args.prefetch_factor if optimal_workers > 0 else 2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and optimal_workers > 0,
        prefetch_factor=args.prefetch_factor if optimal_workers > 0 else 2,
    )

    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=optimal_workers,
            pin_memory=True,
            persistent_workers=args.persistent_workers and optimal_workers > 0,
            prefetch_factor=args.prefetch_factor if optimal_workers > 0 else 2,
        )

    logger.info(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"OOD test dataset: {len(test_dataset)} samples")

    target_size = train_dataset.target_size
    frame_window = train_dataset.clip_length
    input_size = (3, frame_window, target_size[0], target_size[1])

    pretrained_path = None if args.disable_pretrained else args.pretrained_model
    model = ActionRecognitionModel(
        num_classes=args.num_classes,
        pretrained_encoder_path=pretrained_path,
        freeze_encoder=False,
        input_size=input_size,
        use_attentive_classifier=args.use_attentive_classifier,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    ).to(args.device)

    logger.info(
        "Random initialization"
        if args.disable_pretrained
        else f"Pretrained: {args.pretrained_model}"
    )
    param_counts = model.count_trainable_parameters()
    logger.info(
        f"Parameters — Encoder: {param_counts['encoder']:,} | "
        f"Classifier: {param_counts['classifier']:,} | "
        f"Total: {param_counts['total']:,}"
    )

    optimizer = optim.AdamW(
        [
            {
                "params": model.classifier.parameters(),
                "lr": args.classifier_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": model.encoder.parameters(),
                "lr": args.encoder_start_lr,
                "weight_decay": args.weight_decay,
            },
        ],
        eps=1e-8,
    )

    lr_scheduler = ProgressiveLRScheduler(
        optimizer=optimizer,
        classifier_lr=args.classifier_lr,
        encoder_start_lr=args.encoder_start_lr,
        encoder_target_lr=args.encoder_target_lr,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        schedule=args.lr_schedule,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda") if args.use_amp else None

    train_metrics = {"loss": [], "accuracy": []}
    val_metrics = {"loss": [], "accuracy": []}
    test_metrics = {"loss": [], "accuracy": [], "epochs": []} if test_loader else None
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(args.num_epochs):
        classifier_lr, encoder_lr = lr_scheduler.step()

        if epoch < args.warmup_epochs:
            warmup_pct = epoch / args.warmup_epochs * 100
            logger.info(
                f"Epoch {epoch + 1}: Classifier LR={classifier_lr:.2e}, "
                f"Encoder LR={encoder_lr:.2e} (Warmup {warmup_pct:.1f}%)"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}: Classifier LR={classifier_lr:.2e}, Encoder LR={encoder_lr:.2e}"
            )

        train_epoch_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch + 1,
            logger=logger,
            scaler=scaler,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )

        should_save_outputs = args.save_val_outputs and (
            (epoch + 1) % args.save_outputs_every_n_epochs == 0
            or (epoch + 1) == args.num_epochs
        )

        val_epoch_metrics = val_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=args.device,
            epoch=epoch + 1,
            logger=logger,
            use_amp=args.use_amp,
            save_outputs=should_save_outputs,
            output_dir=output_dir,
            segments=val_segments,
        )

        for metric in train_metrics:
            train_metrics[metric].append(train_epoch_metrics[metric])
            val_metrics[metric].append(val_epoch_metrics[metric])

        if test_loader and (epoch + 1) % args.test_every_n_epochs == 0:
            test_epoch_metrics = test_epoch(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=args.device,
                epoch=epoch + 1,
                logger=logger,
                use_amp=args.use_amp,
                save_outputs=args.save_val_outputs,
                output_dir=output_dir,
                segments=test_segments,
            )
            test_metrics["loss"].append(test_epoch_metrics["loss"])
            test_metrics["accuracy"].append(test_epoch_metrics["accuracy"])
            test_metrics["epochs"].append(epoch)
            if test_epoch_metrics["accuracy"] > best_test_acc:
                best_test_acc = test_epoch_metrics["accuracy"]

        if val_epoch_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_epoch_metrics["accuracy"]
            save_best_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=best_val_acc,
                test_accuracy=best_test_acc if test_dataset else None,
                num_classes=args.num_classes,
                args=vars(args),
                filepath=output_dir / "best_model.pt",
            )
            logger.info(f"Saved best model — val accuracy: {best_val_acc:.2f}%")

            if args.save_val_outputs and not should_save_outputs:
                try:
                    save_validation_outputs(
                        predictions=val_epoch_metrics["predictions"],
                        labels=val_epoch_metrics["labels"],
                        logits=val_epoch_metrics["logits"],
                        segments=val_segments,
                        epoch=epoch + 1,
                        output_dir=output_dir,
                        split="val_best",
                    )
                except Exception as e:
                    logger.warning(f"Failed to save best model val outputs: {e}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                args=vars(args),
                filepath=output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
            )
            plot_training_curves(
                train_metrics, val_metrics, output_dir, args.warmup_epochs, test_metrics
            )
            logger.info(f"Checkpoint saved at epoch {epoch + 1}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    plot_training_curves(
        train_metrics, val_metrics, output_dir, args.warmup_epochs, test_metrics
    )

    if test_loader:
        domain_gap = best_val_acc - best_test_acc
        logger.info(f"Domain gap (Val - OOD Test): {domain_gap:.2f}%")


if __name__ == "__main__":
    main()
