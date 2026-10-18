import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tasdf.models.final_model import TASDFNet
from tasdf.data import build_dataset
from tasdf.utils import resolve_data_config


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm tra luồng chạy TASDFNet từ cấu hình")
    parser.add_argument("--config", type=str, default="configs/coco_blastocyst.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--data_dir", type=str, default=None, help="Ghi đè đường dẫn thư mục dữ liệu (tùy chọn)")
    parser.add_argument("--output", type=str, default="visualize/img/model_output_sanity.png", help="Đường dẫn lưu ảnh")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_dir:
        data_cfg = {"type": "folder", "path": args.data_dir, "image_size": 256}
    else:
        data_cfg = resolve_data_config(args.config)

    dataset = build_dataset(data_cfg, split="train", is_labeled=False, is_train=False)
    if len(dataset) == 0:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong cấu hình: {data_cfg}")

    img_tensor = dataset[0].unsqueeze(0).to(device)
    model = TASDFNet(in_channels=1, base_channels=64, num_classes=5, k_iters=10).to(device)
    model.eval()

    with torch.no_grad():
        p1, t1 = model(img_tensor, branch="net1")
        probs = F.softmax(p1, dim=1)

    img_np = img_tensor.squeeze().cpu().numpy()
    skeleton_np = t1[0].mean(dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("1. Ảnh Phôi Đầu Vào")
    axes[0].axis("off")

    titles = ["2. Kênh ZP", "3. Kênh TE", "4. Kênh ICM", "5. Khung xương ITGM"]
    cmaps = ["magma", "viridis", "plasma", "inferno"]
    maps = [
        probs[0, 1].cpu().numpy(),
        probs[0, 2].cpu().numpy(),
        probs[0, 3].cpu().numpy(),
        skeleton_np,
    ]

    for idx in range(4):
        axes[idx + 1].imshow(maps[idx], cmap=cmaps[idx])
        axes[idx + 1].set_title(titles[idx])
        axes[idx + 1].axis("off")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f">> Đã lưu ảnh kiểm tra tại: {out_path.resolve()}")


if __name__ == "__main__":
    main()