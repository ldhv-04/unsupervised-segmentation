from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transforms import EmbryoTransformPipeline


class BaseEmbryoDataset(Dataset, ABC):
    """
    Interface cơ sở cho tất cả các loại Dataset phôi học trong TASDF.
    Hỗ trợ tiền xử lý và tăng cường dữ liệu điều khiển qua config.
    """

    def __init__(
        self,
        size: int = 256,
        is_labeled: bool = True,
        aug_cfg: dict[str, Any] | None = None,
        is_train: bool = False,
    ):
        self.size = size
        self.is_labeled = is_labeled
        self.transform = EmbryoTransformPipeline(size=size, cfg=aug_cfg, is_train=is_train)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def load_raw_grayscale(self, img_path: str) -> np.ndarray:
        """Đọc ảnh xám gốc dưới dạng mảng uint8 [H, W]."""
        img = Image.open(img_path).convert("L")
        return np.array(img, dtype=np.uint8)

    def preprocess_pair(
        self, img_path: str, mask_np: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Áp dụng pipeline biến đổi đồng bộ cho ảnh và mặt nạ."""
        raw_img = self.load_raw_grayscale(img_path)
        img_float, mask_out = self.transform(raw_img, mask_np)
        img_tensor = torch.tensor(img_float, dtype=torch.float32).unsqueeze(0)
        if mask_out is None:
            return img_tensor, None
        return img_tensor, torch.tensor(mask_out, dtype=torch.long)

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        """Đọc và chuẩn hóa một ảnh đơn lẻ (dùng cho nhánh không nhãn)."""
        img_tensor, _ = self.preprocess_pair(img_path, mask_np=None)
        return img_tensor