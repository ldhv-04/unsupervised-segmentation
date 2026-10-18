import os
from pathlib import Path
import shutil
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tasdf.losses import MainLoss
from tasdf.utils.metrics import (
    get_model_parameter_stats,
    format_parameter_summary,
)
from .logger import DualLogger


class Trainer:
    """Trình điều phối huấn luyện bán giám sát TASDF."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict[str, Any],
        logger: DualLogger | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        train_cfg = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})
        model_cfg = cfg.get("model", {})

        req_device = train_cfg.get("device")
        if req_device:
            self.device = torch.device(req_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.use_amp = bool(train_cfg.get("amp", True) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 5e-5)),
        )

        # Đồng bộ max_epochs giữa loss và training
        configured_max_epochs = loss_cfg.get("max_epochs")
        total_train_epochs = int(train_cfg.get("epochs", 2000))
        effective_max_epochs = int(configured_max_epochs) if configured_max_epochs is not None else total_train_epochs

        self.criterion = MainLoss(
            lambda_topo=float(loss_cfg.get("lambda_topo", 0.10)),
            gamma=float(loss_cfg.get("gamma", 0.5)),
            alpha=float(loss_cfg.get("alpha", 1.0)),
            max_epochs=effective_max_epochs,
            omega_0=float(loss_cfg.get("omega_0", 1.0)),
            omega_1=float(loss_cfg.get("omega_1", 1.0)),
            topo_sup_weight=float(loss_cfg.get("topo_sup_weight", 0.10)),
            k_iters=int(model_cfg.get("k_iters", 10)),
        ).to(self.device)

        ckpt_path = checkpoint_dir or train_cfg.get("checkpoint_dir", "checkpoints")
        self.checkpoint_dir = Path(ckpt_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        configured_dir = train_cfg.get("checkpoint_dir")
        self.root_checkpoint_dir = Path(configured_dir) if configured_dir else Path("checkpoints")

        self.logger = logger or DualLogger(self.checkpoint_dir.parent / "logs")

        self.start_epoch = 1
        self.best_val_dice = 0.0
        self.accum_steps = int(train_cfg.get("accum_steps", 1))
        self.log_interval = int(train_cfg.get("log_interval", 20))

    def save_checkpoint(self, epoch: int, is_best: bool = False, custom_name: str | None = None):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": self.cfg,
        }

        last_file = self.checkpoint_dir / "last.pt"
        torch.save(state, str(last_file))

        for alias in ["latest.pt", "checkpoint_latest.pt"]:
            alias_path = self.checkpoint_dir / alias
            if alias_path.exists() or alias_path.is_symlink():
                alias_path.unlink()
            try:
                os.link(str(last_file), str(alias_path))
            except OSError:
                shutil.copyfile(str(last_file), str(alias_path))

        best_file = self.checkpoint_dir / "best.pt"
        if is_best or not best_file.exists():
            torch.save(state, str(best_file))
            alias_best = self.checkpoint_dir / "checkpoint_best.pt"
            if alias_best.exists() or alias_best.is_symlink():
                alias_best.unlink()
            try:
                os.link(str(best_file), str(alias_best))
            except OSError:
                shutil.copyfile(str(best_file), str(alias_best))

        target_dirs = []
        if hasattr(self, "root_checkpoint_dir") and self.root_checkpoint_dir:
            if self.root_checkpoint_dir.resolve() != self.checkpoint_dir.resolve():
                target_dirs.append(self.root_checkpoint_dir)

        sibling_name = "checkpoints" if self.checkpoint_dir.name == "weights" else "weights"
        sibling_dir = self.checkpoint_dir.parent / sibling_name
        if sibling_dir.resolve() != self.checkpoint_dir.resolve():
            target_dirs.append(sibling_dir)

        for out_dir in target_dirs:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(last_file), str(out_dir / "last.pt"))
                if best_file.exists():
                    shutil.copyfile(str(best_file), str(out_dir / "best.pt"))
            except Exception:
                pass

        if custom_name:
            custom_file = self.checkpoint_dir / custom_name
            torch.save(state, str(custom_file))

    def load_checkpoint(self, checkpoint_path: str | Path, resume: bool = True):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if resume:
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and self.use_amp:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_val_dice = ckpt.get("best_val_dice", 0.0)
            self.logger.log_message(
                f">> Tiếp tục huấn luyện từ Epoch {self.start_epoch} (Best Dice trước đó: {self.best_val_dice * 100:.2f}%)"
            )
        else:
            self.logger.log_message(f">> Đã nạp thành công trọng số mô hình từ {path}")

    def train_one_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        labeled_iter = iter(labeled_loader)
        total_loss_accum, sup_accum, distill_accum, steps = 0.0, 0.0, 0.0, 0
        total_steps = len(unlabeled_loader)

        self.optimizer.zero_grad()

        for step, x_u in enumerate(unlabeled_loader, 1):
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            if isinstance(x_u, (tuple, list)):
                x_u = x_u[0]

            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u = x_u.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                (p1_l, t1_l), (p2_l, t2_l) = self.model(x_l, branch="both")
                (p1_u, t1_u), (p2_u, t2_u) = self.model(x_u, branch="both")

                loss, metrics = self.criterion(
                    p1_l,
                    p2_l,
                    y_l,
                    p1_u,
                    p2_u,
                    t1_u,
                    t2_u,
                    current_epoch=epoch,
                    t1_labeled=t1_l,
                    t2_labeled=t2_l,
                )
                loss_step = loss / self.accum_steps

            self.scaler.scale(loss_step).backward()

            if step % self.accum_steps == 0 or step == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            distill_val = metrics["loss_pc"] + self.criterion.lambda_topo * metrics["loss_tc"]
            total_loss_accum += metrics["total_loss"]
            sup_accum += metrics["loss_sup"]
            distill_accum += distill_val
            steps += 1

        return {
            "total_loss": total_loss_accum / max(1, steps),
            "loss_sup": sup_accum / max(1, steps),
            "loss_distill": distill_accum / max(1, steps),
        }

    def train(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader | None = None,
        validator: Any | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        total_epochs = int(epochs or self.cfg.get("training", {}).get("epochs", 2000))
        # Đồng bộ max_epochs của criterion nếu chưa bị khóa cứng
        if self.cfg.get("loss", {}).get("max_epochs") is None:
            self.criterion.max_epochs = total_epochs

        param_stats = get_model_parameter_stats(self.model)
        self.logger.log_message("\n" + format_parameter_summary(param_stats))

        self.logger.log_message(
            f"Bắt đầu huấn luyện: Epoch {self.start_epoch} -> {total_epochs} | Thiết bị: {self.device} | AMP: {self.use_amp}"
        )
        self.logger.log_header()

        for epoch in range(self.start_epoch, total_epochs + 1):
            train_metrics = self.train_one_epoch(labeled_loader, unlabeled_loader, epoch)

            val_mAcc, val_mRec, val_mDice, val_mIoU = 0.0, 0.0, 0.0, 0.0
            per_class_dice = {}
            if validator:
                val_res = validator.validate(self.model)
                val_mAcc = val_res.get("mAccuracy", 0.0)
                val_mRec = val_res.get("mRecall", 0.0)
                val_mDice = val_res.get("mDice", 0.0)
                val_mIoU = val_res.get("mIoU", 0.0)
                per_class_dice = val_res.get("per_class_dice", {})

            epoch_data = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train_total_loss": train_metrics["total_loss"],
                "train_loss_sup": train_metrics["loss_sup"],
                "train_loss_distill": train_metrics["loss_distill"],
                "val_mAccuracy": val_mAcc,
                "val_mRecall": val_mRec,
                "val_mDice": val_mDice,
                "val_mIoU": val_mIoU,
                "per_class_dice": per_class_dice,
            }
            self.logger.log_epoch(epoch_data)

            is_best = (val_mDice > self.best_val_dice) or (epoch == self.start_epoch)
            if val_mDice > self.best_val_dice:
                self.best_val_dice = val_mDice
            self.save_checkpoint(epoch, is_best=is_best)

        best_ckpt = self.checkpoint_dir / "best.pt"
        last_ckpt = self.checkpoint_dir / "last.pt"
        if not best_ckpt.exists() and last_ckpt.exists():
            shutil.copyfile(str(last_ckpt), str(best_ckpt))

        return {
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
            "latest_checkpoint": str(last_ckpt),
            "metrics_path": str(self.logger.json_file),
            "best_val_dice": self.best_val_dice,
        }