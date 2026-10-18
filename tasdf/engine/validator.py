from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tasdf.utils.metrics import compute_sample_metrics, DEFAULT_CLASS_NAMES


class Validator:
    """
    Động cơ đánh giá độc lập mô hình trên tập kiểm tra/kiểm định:
    Tính toán đầy đủ 4 chỉ số theo bài báo TASDF: Accuracy, Recall, Dice, và Jaccard (IoU).
    Tự động loại trừ các lớp không tồn tại trong ground-truth khỏi phép tính trung bình.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        device: torch.device | None = None,
        class_names: dict[int, str] | None = None,
        num_classes: int = 5,
        branch: str = "net1",
    ):
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.num_classes = num_classes
        self.branch = branch

    def validate(self, model: nn.Module) -> dict[str, Any]:
        """Thực hiện đánh giá trên toàn bộ val_loader."""
        model.eval()
        accum = {
            c: {"accuracy": [], "recall": [], "dice": [], "iou": [], "present_count": 0}
            for c in range(1, self.num_classes)
        }

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                if self.branch == "ensemble":
                    (p1, _), (p2, _) = model(x, branch="both")
                    preds = (torch.softmax(p1, dim=1) + torch.softmax(p2, dim=1)) / 2.0
                else:
                    p, _ = model(x, branch=self.branch)
                    preds = torch.softmax(p, dim=1)

                pred_masks = torch.argmax(preds, dim=1).cpu().numpy()
                target_masks = y.numpy()

                for i in range(len(x)):
                    sample_m = compute_sample_metrics(
                        pred_masks[i], target_masks[i], num_classes=self.num_classes
                    )
                    for c in range(1, self.num_classes):
                        if sample_m[c]["present"]:
                            accum[c]["present_count"] += 1
                        if sample_m[c]["active"]:
                            accum[c]["accuracy"].append(sample_m[c]["accuracy"])
                            accum[c]["recall"].append(sample_m[c]["recall"])
                            accum[c]["dice"].append(sample_m[c]["dice"])
                            accum[c]["iou"].append(sample_m[c]["iou"])

        per_class_accuracy = {}
        per_class_recall = {}
        per_class_dice = {}
        per_class_iou = {}

        valid_classes = [c for c in range(1, self.num_classes) if accum[c]["present_count"] > 0]
        if not valid_classes:
            valid_classes = [c for c in range(1, self.num_classes) if len(accum[c]["dice"]) > 0]

        for c in valid_classes:
            name = self.class_names.get(c, DEFAULT_CLASS_NAMES.get(c, f"Class {c}"))
            per_class_accuracy[name] = float(np.mean(accum[c]["accuracy"])) if accum[c]["accuracy"] else 0.0
            per_class_recall[name] = float(np.mean(accum[c]["recall"])) if accum[c]["recall"] else 0.0
            per_class_dice[name] = float(np.mean(accum[c]["dice"])) if accum[c]["dice"] else 0.0
            per_class_iou[name] = float(np.mean(accum[c]["iou"])) if accum[c]["iou"] else 0.0

        mean_acc = float(np.mean(list(per_class_accuracy.values()))) if per_class_accuracy else 0.0
        mean_rec = float(np.mean(list(per_class_recall.values()))) if per_class_recall else 0.0
        mean_dice = float(np.mean(list(per_class_dice.values()))) if per_class_dice else 0.0
        mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

        return {
            "mAccuracy": mean_acc,
            "mRecall": mean_rec,
            "mDice": mean_dice,
            "mIoU": mean_iou,
            "per_class_accuracy": per_class_accuracy,
            "per_class_recall": per_class_recall,
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
        }

    def print_results(self, results: dict[str, Any], title: str = "VAL"):
        """In bảng kết quả đánh giá gồm 4 độ đo chuẩn của bài báo."""
        print("\n" + "=" * 88)
        print(f"  KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP | TẬP: [{title.upper()}] | NHÁNH: [{self.branch}]")
        print("=" * 88)
        print(
            f"{'Cấu trúc Phôi học':<26} | {'Accuracy (%)':<13} | {'Recall (%)':<12} | "
            f"{'Dice (%)':<12} | {'Jaccard/IoU (%)':<15}"
        )
        print("-" * 88)
        for name, dice in results["per_class_dice"].items():
            acc = results["per_class_accuracy"].get(name, 0.0)
            rec = results["per_class_recall"].get(name, 0.0)
            iou = results["per_class_iou"].get(name, 0.0)
            print(
                f"{name:<26} | {acc * 100:<13.2f} | {rec * 100:<12.2f} | "
                f"{dice * 100:<12.2f} | {iou * 100:<15.2f}"
            )
        print("-" * 88)
        print(
            f"{'TRUNG BÌNH (MEAN)':<26} | {results['mAccuracy'] * 100:<13.2f} | "
            f"{results['mRecall'] * 100:<12.2f} | {results['mDice'] * 100:<12.2f} | "
            f"{results['mIoU'] * 100:<15.2f}"
        )
        print("=" * 88 + "\n")