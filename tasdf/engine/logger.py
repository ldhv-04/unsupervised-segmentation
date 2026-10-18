from datetime import datetime
import json
from pathlib import Path
from typing import Any
import torch


class DualLogger:
    """Hệ thống ghi nhận kép Text stream & JSON stream."""

    def __init__(self, output_dir: str | Path, exp_name: str = "train"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "train.log"
        self.json_file = self.output_dir / "metrics.json"
        self.history: list[dict[str, Any]] = []

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 105}\n")
            f.write(f">> PHIÊN HUẤN LUYỆN TASDF: {exp_name.upper()} | KHỞI TẠO: {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 105}\n")

    def log_header(self):
        header = (
            f"{'Epoch':<8} | {'GPU_Mem':<8} | {'L_total':<8} | {'L_sup':<8} | "
            f"{'L_dist':<8} | {'Val_Acc':<8} | {'Val_Rec':<8} | {'Val_Dice':<9} | "
            f"{'Val_IoU':<8} | {'ZP_Dice':<8} | {'TE_Dice':<8} | {'ICM_Dice':<8} | {'BC_Dice':<8}"
        )
        separator = "-" * len(header)
        print("\n" + separator)
        print(header)
        print(separator)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(separator + "\n" + header + "\n" + separator + "\n")

    def _fmt_class(self, per_class: dict[str, float], *candidates: str) -> str:
        # 1. Khớp chính xác
        for key in candidates:
            if key in per_class and per_class[key] is not None:
                return f"{per_class[key] * 100:.1f}%"
        # 2. Khớp từ khóa con không phân biệt hoa thường
        for k_dict, val in per_class.items():
            k_upper = str(k_dict).upper()
            for cand in candidates:
                if cand.upper() in k_upper and val is not None:
                    return f"{val * 100:.1f}%"
        return "N/A"

    def log_epoch(self, metrics: dict[str, Any]):
        epoch = metrics.get("epoch", 0)
        total_epochs = metrics.get("total_epochs", 0)
        epoch_str = f"{epoch:02d}/{total_epochs:02d}" if total_epochs else f"{epoch:02d}"

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_str = f"{gpu_mem:.2f}G"

        l_total = f"{metrics.get('train_total_loss', 0.0):.4f}"
        l_sup = f"{metrics.get('train_loss_sup', 0.0):.4f}"
        l_dist = f"{metrics.get('train_loss_distill', 0.0):.4f}"

        val_acc = f"{metrics.get('val_mAccuracy', 0.0) * 100:.2f}%"
        val_rec = f"{metrics.get('val_mRecall', 0.0) * 100:.2f}%"
        val_dice = f"{metrics.get('val_mDice', 0.0) * 100:.2f}%"
        val_iou = f"{metrics.get('val_mIoU', 0.0) * 100:.2f}%"

        per_class = metrics.get("per_class_dice", {})
        zp = self._fmt_class(per_class, "Zona Pellucida (ZP)", "ZP", "Zona")
        te = self._fmt_class(per_class, "Trophectoderm (TE)", "TE", "Trophectoderm")
        icm = self._fmt_class(per_class, "Inner Cell Mass (ICM)", "ICM", "Inner")
        bc = self._fmt_class(per_class, "Blastocyst Cavity (BC)", "BC", "Blastocoel", "Cavity")

        row = (
            f"{epoch_str:<8} | {mem_str:<8} | {l_total:<8} | {l_sup:<8} | "
            f"{l_dist:<8} | {val_acc:<8} | {val_rec:<8} | {val_dice:<9} | "
            f"{val_iou:<8} | {zp:<8} | {te:<8} | {icm:<8} | {bc:<8}"
        )
        print(row)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(row + "\n")

        record = metrics.copy()
        record["timestamp"] = datetime.now().isoformat()
        record["gpu_mem_gb"] = gpu_mem
        self.history.append(record)

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")