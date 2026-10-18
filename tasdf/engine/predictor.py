from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from tasdf.utils.metrics import (
    colorize_mask,
    DEFAULT_CLASS_NAMES,
    DEFAULT_PALETTE,
)


class Predictor:
    """
    Động cơ suy luận và trực quan hóa phân đoạn phôi học thuần túy từ dự đoán của mạng TASDF.
    """

    def __init__(
        self,
        image_size: int = 256,
        device: torch.device | None = None,
        palette: dict[int, tuple[int, int, int]] | None = None,
        class_names: dict[int, str] | None = None,
        branch: str = "net1",
    ):
        self.image_size = image_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.palette = palette or DEFAULT_PALETTE
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.branch = branch

    def predict_single(
        self, model: nn.Module, img_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Suy luận trực tiếp trên một ảnh duy nhất."""
        path = Path(img_path)
        orig_img = Image.open(path).convert("RGB")
        orig_w, orig_h = orig_img.size
        orig_np = np.array(orig_img)

        gray_resized = orig_img.convert("L").resize((self.image_size, self.image_size))
        img_tensor = (
            torch.tensor(np.array(gray_resized, dtype=np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        model.eval()
        with torch.no_grad():
            if self.branch == "ensemble":
                (p1, _), (p2, _) = model(img_tensor, branch="both")
                probs = (torch.softmax(p1, dim=1) + torch.softmax(p2, dim=1)) / 2.0
            else:
                p, _ = model(img_tensor, branch=self.branch)
                probs = torch.softmax(p, dim=1)

            pred_small = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_orig = cv2.resize(
            pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        return orig_np, pred_orig

    def save_results(
        self, orig_rgb: np.ndarray, pred_mask: np.ndarray, out_dir: Path, stem: str
    ) -> dict[str, str]:
        """Lưu mặt nạ số, ảnh phủ màu và ảnh đối sánh."""
        out_dir.mkdir(parents=True, exist_ok=True)
        color_mask = colorize_mask(pred_mask, self.palette)

        overlay = cv2.addWeighted(orig_rgb, 0.65, color_mask, 0.35, 0)
        composite = np.hstack([orig_rgb, color_mask, overlay])

        paths = {
            "mask_raw": str(out_dir / f"{stem}_mask_raw.png"),
            "overlay": str(out_dir / f"{stem}_overlay.png"),
            "composite": str(out_dir / f"{stem}_composite.png"),
        }
        cv2.imwrite(paths["mask_raw"], pred_mask)
        cv2.imwrite(paths["overlay"], cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(paths["composite"], cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        return paths

    def predict(
        self,
        model: nn.Module,
        source: str | Path | list[Path],
        output_dir: str | Path = "runs/predict",
        save: bool = True,
    ) -> list[dict[str, Any]]:
        """Chạy dự đoán trên nguồn dữ liệu (file ảnh đơn, danh sách hoặc thư mục)."""
        model = model.to(self.device)
        out_p = Path(output_dir)

        if isinstance(source, (str, Path)):
            src_p = Path(source)
            if src_p.is_file():
                files = [src_p]
            elif src_p.is_dir():
                files = sorted(list(src_p.glob("*.jpg")) + list(src_p.glob("*.png")))
            else:
                raise FileNotFoundError(f"Không tìm thấy đường dẫn nguồn: {source}")
        else:
            files = [Path(f) for f in source]

        results = []
        for f in files:
            orig_rgb, pred_mask = self.predict_single(model, f)
            counts = {
                self.class_names.get(int(c), str(c)): int((pred_mask == c).sum())
                for c in np.unique(pred_mask)
                if c > 0
            }

            res_item: dict[str, Any] = {
                "file": str(f),
                "stem": f.stem,
                "counts": counts,
                "mask": pred_mask,
            }

            if save:
                saved_paths = self.save_results(orig_rgb, pred_mask, out_p, f.stem)
                res_item["saved_paths"] = saved_paths

            results.append(res_item)
            print(f"   [+] {f.stem}: {counts if counts else 'Background only'}")

        return results