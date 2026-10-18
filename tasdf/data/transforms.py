import random
from typing import Any
import cv2
import numpy as np


class EmbryoTransformPipeline:
    """
    Quy trình tiền xử lý khử nhiễu và tăng cường dữ liệu 5 bước theo bài báo TASDF (Mục 4.2):
    1. Random Rotation (Xoay ngẫu nhiên đồng bộ ảnh & mask)
    2. Random Scale Crop (Co giãn tỷ lệ và cắt ngẫu nhiên đồng bộ ảnh & mask)
    3. Random Horizontal Flip (Lật ngang ngẫu nhiên đồng bộ ảnh & mask)
    4. Color Jitter (Biến đổi độ sáng và độ tương phản trên ảnh)
    5. Random Gaussian Blur (Làm mờ Gaussian ngẫu nhiên trên ảnh)
    """

    def __init__(self, size: int = 256, cfg: dict[str, Any] | None = None, is_train: bool = True):
        self.size = size
        self.cfg = cfg or {}
        self.is_train = is_train
        self.enabled = bool(self.cfg.get("enabled", True)) and is_train

        self.denoise_cfg = self.cfg.get("denoise", {})
        self.denoise_enabled = bool(self.denoise_cfg.get("enabled", False))

    def apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """Khử nhiễu tiền xử lý cho ảnh kính hiển vi HMC (áp dụng trên ảnh uint8 [0, 255])."""
        if not self.denoise_enabled:
            return image

        method = str(self.denoise_cfg.get("method", "median")).lower()
        ksize = int(self.denoise_cfg.get("kernel_size", 3))
        if ksize % 2 == 0:
            ksize += 1

        if method == "median":
            return cv2.medianBlur(image, ksize)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        return image

    def _random_rotation(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        max_deg = float(self.cfg.get("max_rotation_deg", 30.0))
        angle = random.uniform(-max_deg, max_deg)
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_rot = cv2.warpAffine(
            img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        mask_rot = None
        if mask is not None:
            mask_rot = cv2.warpAffine(
                mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        return img_rot, mask_rot

    def _random_scale_crop(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        scale_range = self.cfg.get("scale_range", [0.85, 1.15])
        scale = random.uniform(float(scale_range[0]), float(scale_range[1]))
        h, w = img.shape[:2]
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = (
            cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if mask is not None
            else None
        )

        if new_h >= self.size and new_w >= self.size:
            top = random.randint(0, new_h - self.size)
            left = random.randint(0, new_w - self.size)
            img_cropped = img_scaled[top : top + self.size, left : left + self.size]
            mask_cropped = (
                mask_scaled[top : top + self.size, left : left + self.size]
                if mask_scaled is not None
                else None
            )
        else:
            pad_h = max(0, self.size - new_h)
            pad_w = max(0, self.size - new_w)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            img_cropped = cv2.copyMakeBorder(
                img_scaled, top, bottom, left, right, cv2.BORDER_REFLECT_101
            )
            mask_cropped = (
                cv2.copyMakeBorder(
                    mask_scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
                )
                if mask_scaled is not None
                else None
            )

        return img_cropped, mask_cropped

    def _random_horizontal_flip(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        prob = float(self.cfg.get("flip_prob", 0.5))
        if random.random() < prob:
            img = np.fliplr(img).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        return img, mask

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        brightness_delta = float(self.cfg.get("brightness_delta", 0.15))
        contrast_range = self.cfg.get("contrast_range", [0.85, 1.15])

        alpha = random.uniform(float(contrast_range[0]), float(contrast_range[1]))
        beta = random.uniform(-brightness_delta, brightness_delta)
        img_out = img * alpha + beta
        return np.clip(img_out, 0.0, 1.0)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        prob = float(self.cfg.get("blur_prob", 0.3))
        if random.random() < prob:
            ksize = random.choice([3, 5])
            sigma = random.uniform(0.1, 1.5)
            img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
        return img

    def __call__(
        self, image_uint8: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # 1. Tiền xử lý khử nhiễu (nếu bật)
        img = self.apply_denoise(image_uint8)

        # 2. Đưa về kích thước chuẩn trước khi biến đổi
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # 3. Các phép biến đổi hình học trên tập train (nếu bật)
        if self.enabled:
            if self.cfg.get("random_rotation", True):
                img, mask = self._random_rotation(img, mask)
            if self.cfg.get("random_scale_crop", True):
                img, mask = self._random_scale_crop(img, mask)
            if self.cfg.get("random_horizontal_flip", True):
                img, mask = self._random_horizontal_flip(img, mask)

        # 4. Chuẩn hóa về [0.0, 1.0]
        img_float = img.astype(np.float32) / 255.0

        # 5. Các phép biến đổi quang học trên tập train (nếu bật)
        if self.enabled:
            if self.cfg.get("color_jitter", True):
                img_float = self._color_jitter(img_float)
            if self.cfg.get("gaussian_blur", True):
                img_float = self._gaussian_blur(img_float)

        img_float = np.clip(img_float, 0.0, 1.0).astype(np.float32)
        return img_float, mask