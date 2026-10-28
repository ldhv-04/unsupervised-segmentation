import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from .base import BaseEmbryoDataset


class COCOEmbryoDataset(BaseEmbryoDataset):
    """
    Bộ nạp dữ liệu phôi nang từ định dạng COCO Segmentation JSON.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        size: int = 256,
        is_labeled: bool = True,
        class_mapping: dict[str, int] | None = None,
        aug_cfg: dict[str, Any] | None = None,
        is_train: bool | None = None,
    ):
        if is_train is None:
            is_train = split.lower() == "train"
        super().__init__(size=size, is_labeled=is_labeled, aug_cfg=aug_cfg, is_train=is_train)
        self.root_path = Path(root_dir)
        self.split = split

        candidates = [
            self.root_path / split / "_annotations.coco.json",
            self.root_path / split / "annotations.json",
            self.root_path / f"annotations/{split}.json",
            self.root_path / "_annotations.coco.json",
            self.root_path / "annotations.json",
        ]
        json_path = None
        for c in candidates:
            if c.exists():
                json_path = c
                break

        if not json_path:
            raise FileNotFoundError(
                f"Không tìm thấy file nhãn COCO trong '{self.root_path}' cho tập '{split}'."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            self.coco = json.load(f)

        self.img_dir = json_path.parent
        self.images = self.coco.get("images", [])
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.coco.get("categories", [])}

        self.img_id_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in self.coco.get("annotations", []):
            if ann.get("segmentation"):
                self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.class_mapping = class_mapping

        # Phân loại rõ chỉ số ảnh có nhãn và không nhãn thực tế
        self.labeled_indices: list[int] = []
        self.unlabeled_indices: list[int] = []
        for idx, img_info in enumerate(self.images):
            if len(self.img_id_to_anns.get(img_info["id"], [])) > 0:
                self.labeled_indices.append(idx)
            else:
                self.unlabeled_indices.append(idx)

    def _map_category_to_class_id(self, cat_name: str) -> int:
        if self.class_mapping and cat_name in self.class_mapping:
            return self.class_mapping[cat_name]
        name_upper = cat_name.upper()
        if "ZP" in name_upper or "ZONA" in name_upper:
            return 1
        elif "TE" in name_upper or "TROPHECTODERM" in name_upper:
            return 2
        elif "ICM" in name_upper or "INNER" in name_upper:
            return 3
        elif "BC" in name_upper or "BLASTOCOEL" in name_upper or "CAVITY" in name_upper or "BLASTOCYST" in name_upper:
            return 4
        return 0

    def _rasterize_polygons(self, anns: list, orig_w: int, orig_h: int) -> np.ndarray:
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        def draw_priority(ann):
            cid = self._map_category_to_class_id(self.cat_id_to_name.get(ann["category_id"], ""))
            priority_map = {1: 0, 2: 1, 4: 2, 3: 3}
            area = float(ann.get("area", 0.0))
            return (priority_map.get(cid, 0), -area)

        sorted_anns = sorted(anns, key=draw_priority)
        for ann in sorted_anns:
            cat_name = self.cat_id_to_name.get(ann["category_id"], "")
            class_id = self._map_category_to_class_id(cat_name)
            if class_id <= 0:
                continue
            for seg in ann.get("segmentation", []):
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], color=class_id)
        return mask

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]
        if not img_path.exists():
            img_path = self.root_path / img_info["file_name"]

        if not self.is_labeled:
            return self.preprocess_image(str(img_path))

        anns = self.img_id_to_anns.get(img_info["id"], [])
        if not anns:
            raise ValueError(
                f"Ảnh '{img_info['file_name']}' (id={img_info['id']}) không có nhãn annotation. "
                f"Không được phép nạp mẫu không nhãn vào luồng có giám sát (is_labeled=True)."
            )

        orig_w = img_info.get("width", self.size)
        orig_h = img_info.get("height", self.size)
        mask_np = self._rasterize_polygons(anns, orig_w, orig_h)
        img_tensor, mask_tensor = self.preprocess_pair(str(img_path), mask_np)
        return img_tensor, mask_tensor