from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .coco import COCOEmbryoDataset
from .folder import FolderEmbryoDataset


def build_dataset(
    data_cfg: dict, split: str = "train", is_labeled: bool = True, is_train: bool | None = None
) -> Dataset:
    """Factory khởi tạo Dataset tương ứng với cấu hình và chế độ tăng cường dữ liệu."""
    ds_type = (data_cfg.get("type") or data_cfg.get("format", "coco")).lower()
    root_dir = data_cfg.get("path") or data_cfg.get("root_dir", ".")
    size = int(data_cfg.get("image_size", 256))
    aug_cfg = data_cfg.get("augmentation")

    actual_split = data_cfg.get(split, split)
    if is_train is None:
        is_train = split.lower() == "train"

    if ds_type == "coco":
        return COCOEmbryoDataset(
            root_dir=root_dir,
            split=actual_split,
            size=size,
            is_labeled=is_labeled,
            aug_cfg=aug_cfg,
            is_train=is_train,
        )
    elif ds_type in ("folder", "embryo", "mask"):
        return FolderEmbryoDataset(
            data_dir=root_dir,
            split=actual_split,
            size=size,
            is_labeled=is_labeled,
            aug_cfg=aug_cfg,
            is_train=is_train,
        )
    else:
        raise ValueError(
            f"Định dạng dữ liệu '{ds_type}' chưa được hỗ trợ. Hỗ trợ: ['coco', 'folder']"
        )


def build_dataloaders(
    data_cfg: dict,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo bộ ba DataLoader: (labeled_loader, unlabeled_loader, val_loader).
    Đảm bảo 100% các mẫu trong labeled_loader đều có nhãn thật và không bao giờ
    lấy nhầm mẫu khuyết nhãn vào luồng giám sát.
    """
    full_train = build_dataset(data_cfg, split="train", is_labeled=True, is_train=True)
    unlabeled_train = build_dataset(data_cfg, split="train", is_labeled=False, is_train=True)

    total_samples = len(full_train)
    avail_labeled = getattr(full_train, "labeled_indices", list(range(total_samples)))
    avail_unlabeled = getattr(full_train, "unlabeled_indices", [])

    if len(avail_labeled) == 0:
        raise RuntimeError("Tập huấn luyện không có bất kỳ mẫu nào có nhãn để học bán giám sát!")

    split_mode = str(data_cfg.get("split_mode", "auto")).lower()
    use_physical = bool(data_cfg.get("use_physical_split", False)) or split_mode == "physical"

    # Nếu tập dữ liệu đã được tách sẵn ảnh không nhãn vật lý (như bộ 10% COCO hoặc thư mục thiếu mask)
    # và người dùng chọn 'physical' hoặc 'auto' (khi không ép cắt nhỏ hơn nữa):
    if use_physical or (split_mode == "auto" and len(avail_unlabeled) > 0 and "labeled_ratio" not in data_cfg):
        labeled_indices = list(avail_labeled)
        unlabeled_indices = list(avail_unlabeled)
    else:
        labeled_ratio = float(data_cfg.get("labeled_ratio", 0.10))
        # Nếu bộ dữ liệu đã được lọc vật lý sẵn về đúng tỷ lệ, giữ nguyên tập có nhãn vật lý
        if len(avail_unlabeled) > 0 and len(avail_labeled) <= max(1, round(total_samples * labeled_ratio)):
            labeled_indices = list(avail_labeled)
            unlabeled_indices = list(avail_unlabeled)
        else:
            # Chia từ tập các mẫu thực sự có nhãn (avail_labeled)
            target_num_labeled = max(1, int(round(len(avail_labeled) * labeled_ratio)))
            target_num_labeled = min(target_num_labeled, len(avail_labeled))

            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(avail_labeled), generator=generator).tolist()

            labeled_indices = [avail_labeled[i] for i in perm[:target_num_labeled]]
            remaining_labeled_as_unlabeled = [avail_labeled[i] for i in perm[target_num_labeled:]]
            unlabeled_indices = remaining_labeled_as_unlabeled + list(avail_unlabeled)

            # Trường hợp labeled_ratio = 1.0 (100% labeled) và không có ảnh không nhãn vật lý:
            # dùng toàn bộ ảnh cho cả nhánh chưng cất để duy trì tính nhất quán
            if len(unlabeled_indices) == 0:
                unlabeled_indices = list(labeled_indices)

    pin = pin_memory and torch.cuda.is_available()

    labeled_loader = DataLoader(
        Subset(full_train, labeled_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(labeled_indices) >= batch_size else False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    unlabeled_loader = DataLoader(
        Subset(unlabeled_train, unlabeled_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(unlabeled_indices) >= batch_size else False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    val_split_name = "val" if "val" in data_cfg else "valid"
    val_ds = build_dataset(data_cfg, split=val_split_name, is_labeled=True, is_train=False)
    val_avail_labeled = getattr(val_ds, "labeled_indices", list(range(len(val_ds))))
    val_subset = Subset(val_ds, val_avail_labeled) if len(val_avail_labeled) < len(val_ds) else val_ds

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return labeled_loader, unlabeled_loader, val_loader