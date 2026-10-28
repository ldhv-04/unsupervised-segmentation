from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Subset

from tasdf.models import TASDFNet
from tasdf.engine import Trainer, Validator, Predictor, DualLogger
from tasdf.data import build_dataloaders, build_dataset
from tasdf.utils import (
    DEFAULT_CONFIG,
    deep_merge,
    load_yaml,
    resolve_data_config,
    get_model_parameter_stats,
    format_parameter_summary,
)


class TASDF:
    """Điểm truy cập chính cho việc huấn luyện, đánh giá và suy luận mô hình TASDF."""

    def __init__(
        self,
        model: str | Path | dict | None = None,
        task: str = "segment",
    ):
        self.task = task
        self.cfg = DEFAULT_CONFIG.copy()
        self.checkpoint_info = None

        model_kwargs = self.cfg["model"].copy()

        if model is not None:
            if isinstance(model, dict):
                self.cfg = deep_merge(self.cfg, model)
                model_kwargs = self.cfg.get("model", model_kwargs)
            else:
                m_path = Path(model)
                if m_path.suffix in (".yaml", ".yml"):
                    user_yaml = load_yaml(m_path)
                    self.cfg = deep_merge(self.cfg, user_yaml)
                    model_kwargs = self.cfg.get("model", model_kwargs)
                elif m_path.suffix == ".pt":
                    ckpt = torch.load(str(m_path), map_location="cpu", weights_only=False)
                    if "config" in ckpt:
                        self.cfg = deep_merge(self.cfg, ckpt["config"])
                        model_kwargs = self.cfg.get("model", model_kwargs)
                    self.checkpoint_info = m_path

        self.model = TASDFNet(
            in_channels=model_kwargs.get("in_channels", 1),
            base_channels=model_kwargs.get("base_channels", 64),
            num_classes=model_kwargs.get("num_classes", 5),
            k_iters=model_kwargs.get("k_iters", 10),
        )

        if self.checkpoint_info is not None:
            ckpt = torch.load(str(self.checkpoint_info), map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)

    def _get_unique_exp_dir(self, project: str | Path, name: str) -> Path:
        base = Path(project)
        target = base / name
        if not target.exists():
            return target
        idx = 2
        while (base / f"{name}{idx}").exists():
            idx += 1
        return base / f"{name}{idx}"

    def train(
        self,
        data: str | Path | dict,
        epochs: int = 2000,
        batch: int = 4,
        imgsz: int = 256,
        lr: float = 1e-4,
        device: str | None = None,
        amp: bool = True,
        accum_steps: int = 1,
        project: str = "runs/train",
        name: str = "exp",
        resume: bool = False,
        num_workers: int = 2,
        seed: int = 42,
        augment: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if data is None or str(data).lower() == "none":
            data = self.cfg
        data_cfg = resolve_data_config(data, default_aug=self.cfg.get("augmentation"))
        data_cfg["image_size"] = imgsz
        if augment is not None:
            data_cfg["augmentation"]["enabled"] = bool(augment)

        train_overrides = {
            "training": {
                "epochs": epochs,
                "batch_size": batch,
                "learning_rate": lr,
                "device": device,
                "amp": amp,
                "accum_steps": accum_steps,
                "num_workers": num_workers,
            },
            "loss": {
                "max_epochs": epochs,
            },
        }
        train_overrides["training"].update(kwargs)
        self.cfg = deep_merge(self.cfg, train_overrides)

        exp_dir = self._get_unique_exp_dir(project, name)
        logger = DualLogger(output_dir=exp_dir, exp_name=exp_dir.name)
        ckpt_dir = exp_dir / "checkpoints"

        labeled_loader, unlabeled_loader, val_loader = build_dataloaders(
            data_cfg=data_cfg,
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=True,
            seed=seed,
        )

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validator = Validator(
            val_loader=val_loader,
            device=dev,
            class_names=data_cfg.get("names"),
            num_classes=self.cfg["model"]["num_classes"],
        )

        trainer = Trainer(
            model=self.model,
            cfg=self.cfg,
            logger=logger,
            checkpoint_dir=ckpt_dir,
        )

        if resume and self.checkpoint_info:
            trainer.load_checkpoint(self.checkpoint_info, resume=True)

        results = trainer.train(
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            validator=validator,
            epochs=epochs,
        )
        results["exp_dir"] = str(exp_dir)
        return results

    def val(
        self,
        data: str | Path | dict,
        split: str = "val",
        batch: int = 4,
        imgsz: int = 256,
        device: str | None = None,
        branch: str = "net1",
        **kwargs,
    ) -> dict[str, Any]:
        if data is None or str(data).lower() == "none":
            data = self.cfg
        data_cfg = resolve_data_config(data, default_aug=self.cfg.get("augmentation"))        
        data_cfg["image_size"] = imgsz

        val_ds = build_dataset(data_cfg, split=split, is_labeled=True, is_train=False)
        avail_labeled = getattr(val_ds, "labeled_indices", list(range(len(val_ds))))
        val_subset = Subset(val_ds, avail_labeled) if len(avail_labeled) < len(val_ds) else val_ds
        loader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dev)
        validator = Validator(
            val_loader=loader,
            device=dev,
            class_names=data_cfg.get("names"),
            num_classes=self.cfg["model"]["num_classes"],
            branch=branch,
        )
        res = validator.validate(self.model)
        validator.print_results(res, title=split)
        return res

    def predict(
        self,
        source: str | Path | list[Path],
        imgsz: int = 256,
        save: bool = True,
        output_dir: str | Path = "runs/predict",
        device: str | None = None,
        branch: str = "net1",
        **kwargs,
    ) -> list[dict[str, Any]]:
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dev)
        predictor = Predictor(
            image_size=imgsz,
            device=dev,
            branch=branch,
        )
        return predictor.predict(self.model, source=source, output_dir=output_dir, save=save)

    def info(self, detailed: bool = True) -> dict[str, Any]:
        stats = get_model_parameter_stats(self.model)
        if detailed:
            print(format_parameter_summary(stats))
        return stats