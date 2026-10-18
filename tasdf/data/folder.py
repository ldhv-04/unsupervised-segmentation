from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from .base import BaseEmbryoDataset


class FolderEmbryoDataset(BaseEmbryoDataset):
    """
    Bộ nạp dữ liệu từ thư mục ảnh và mặt nạ rời (Mask PNG).
    Tự động phân tách file có mặt nạ nhãn và file không có mặt nạ nhãn cho học bán giám sát.
    Tuyệt đối không sinh mặt nạ giả toàn 0 khi thiếu file nhãn.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "",
        size: int = 256,
        is_labeled: bool = True,
        aug_cfg: dict[str, Any] | None = None,
        is_train: bool | None = None,
    ):
        if is_train is None:
            is_train = split.lower() in ("train", "")
        super().__init__(size=size, is_labeled=is_labeled, aug_cfg=aug_cfg, is_train=is_train)

        base = Path(data_dir)
        self.folder_path = base / split if split and (base / split).exists() else base

        img_candidates = [self.folder_path / "images", self.folder_path]
        self.img_dir = self.folder_path
        for c in img_candidates:
            if c.exists() and (list(c.glob("*.jpg")) or list(c.glob("*.png"))):
                self.img_dir = c
                break

        self.mask_dir = self.folder_path / "masks"
        has_separate_mask_dir = self.mask_dir.exists() and self.mask_dir.resolve() != self.img_dir.resolve()

        all_files = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )
        # Loại trừ các file có hậu tố _mask.png nếu chung thư mục
        self.img_files = [f for f in all_files if not f.stem.endswith("_mask")]

        self.mask_paths: dict[int, Path] = {}
        self.labeled_indices: list[int] = []
        self.unlabeled_indices: list[int] = []

        for idx, img_p in enumerate(self.img_files):
            mask_p = self._find_mask_path(img_p, has_separate_mask_dir)
            if mask_p is not None:
                self.mask_paths[idx] = mask_p
                self.labeled_indices.append(idx)
            else:
                self.unlabeled_indices.append(idx)

    def _find_mask_path(self, img_p: Path, has_separate_mask_dir: bool) -> Path | None:
        candidates = []
        if has_separate_mask_dir:
            candidates.extend([
                self.mask_dir / f"{img_p.stem}.png",
                self.mask_dir / f"{img_p.stem}_mask.png",
                self.mask_dir / img_p.name,
            ])
        else:
            candidates.append(self.img_dir / f"{img_p.stem}_mask.png")

        for c in candidates:
            if c.exists() and c.resolve() != img_p.resolve():
                return c
        return None

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_p = self.img_files[idx]
        if not self.is_labeled:
            return self.preprocess_image(str(img_p))

        mask_p = self.mask_paths.get(idx)
        if mask_p is None or not mask_p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mặt nạ nhãn cho ảnh '{img_p.name}'. "
                f"Trong mô hình bán giám sát TASDF, ảnh không nhãn phải được đưa vào unlabeled_loader (is_labeled=False)."
            )

        mask_img = Image.open(mask_p)
        mask_np = np.array(mask_img, dtype=np.uint8)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        img_tensor, mask_tensor = self.preprocess_pair(str(img_p), mask_np)
        return img_tensor, mask_tensor