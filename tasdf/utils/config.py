from pathlib import Path
from typing import Any
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "format": "coco",
        "path": "data/blastocyst_coco",
        "train": "train",
        "val": "valid",
        "test": "test",
        "image_size": 256,
        "split_mode": "auto",  # 'auto', 'physical', hoặc 'ratio'
        "labeled_ratio": 0.10,
        "names": {
            1: "Zona Pellucida (ZP)",
            2: "Trophectoderm (TE)",
            3: "Inner Cell Mass (ICM)",
            4: "Blastocyst Cavity (BC)",
        },
    },
    "augmentation": {
        "enabled": True,
        "random_rotation": True,
        "max_rotation_deg": 30.0,
        "random_scale_crop": True,
        "scale_range": [0.85, 1.15],
        "random_horizontal_flip": True,
        "flip_prob": 0.5,
        "color_jitter": True,
        "brightness_delta": 0.15,
        "contrast_range": [0.85, 1.15],
        "gaussian_blur": True,
        "blur_prob": 0.3,
        "denoise": {
            "enabled": False,
            "method": "median",  # 'median', 'gaussian', hoặc 'clahe'
            "kernel_size": 3,
        },
    },
    "model": {
        "in_channels": 1,
        "base_channels": 64,
        "num_classes": 5,
        "k_iters": 10,
    },
    "loss": {
        "omega_0": 1.0,
        "omega_1": 1.0,
        "lambda_topo": 0.10,
        "gamma": 0.5,
        "alpha": 1.0,
        "topo_sup_weight": 0.10,  # Neo khung xương trên tập có nhãn để chống sụp đổ nghiệm
        "max_epochs": None,       # Tự động đồng bộ theo training.epochs nếu để None
    },
    "training": {
        "epochs": 100,
        "batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.00005,
        "amp": True,
        "accum_steps": 1,
        "num_workers": 2,
        "log_interval": 20,
        "save_period": 1,
    },
}


def load_yaml(path_or_str: str | Path) -> dict[str, Any]:
    """Nạp file YAML an toàn."""
    path = Path(path_or_str)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict, update: dict) -> dict:
    """Hợp nhất đệ quy 2 dictionary."""
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_data_config(
    data_input: str | Path | dict,
    default_aug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Chuẩn hóa cấu hình dữ liệu đầu vào, giữ nguyên các thiết lập chia tập và tăng cường dữ liệu.
    """
    if isinstance(data_input, (str, Path)):
        p = Path(data_input)
        raw_cfg = load_yaml(p)
        base_dir = p.parent
    elif isinstance(data_input, dict):
        raw_cfg = data_input.copy()
        base_dir = Path(".")
    else:
        raise TypeError(f"Dữ liệu cấu hình không hợp lệ: {type(data_input)}")

    data_cfg = raw_cfg.get("dataset", raw_cfg)
    aug_cfg = raw_cfg.get("augmentation", data_cfg.get("augmentation", default_aug or DEFAULT_CONFIG["augmentation"]))

    data_path = Path(data_cfg.get("path") or data_cfg.get("root_dir", "."))
    if not data_path.is_absolute():
        data_path = (base_dir / data_path).resolve()

    resolved: dict[str, Any] = {
        "type": (data_cfg.get("format") or data_cfg.get("type", "coco")).lower(),
        "root_dir": str(data_path),
        "train": data_cfg.get("train", "train"),
        "val": data_cfg.get("val") or data_cfg.get("valid", "valid"),
        "test": data_cfg.get("test", "test"),
        "image_size": int(data_cfg.get("image_size", 256)),
        "names": data_cfg.get("names", DEFAULT_CONFIG["dataset"]["names"]),
        "augmentation": deep_merge(DEFAULT_CONFIG["augmentation"], aug_cfg),
    }

    if "labeled_ratio" in data_cfg and data_cfg["labeled_ratio"] is not None:
        resolved["labeled_ratio"] = float(data_cfg["labeled_ratio"])
    if "split_mode" in data_cfg:
        resolved["split_mode"] = str(data_cfg["split_mode"]).lower()
    if "use_physical_split" in data_cfg:
        resolved["use_physical_split"] = bool(data_cfg["use_physical_split"])

    return resolved