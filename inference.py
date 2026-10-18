"""
inference.py - CLI entrypoint for inference and mask visualization on embryo images.
"""
import argparse
from pathlib import Path
from tasdf import TASDF


def parse_args():
    parser = argparse.ArgumentParser(description="TASDF Inference CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="runs/predict", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=256, help="Inference image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--branch", type=str, default="net1", choices=["net1", "net2", "ensemble"], help="Inference branch")
    return parser.parse_args()


def main():
    args = parse_args()
    model = TASDF(model=args.checkpoint)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        device=args.device,
        branch=args.branch,
        save=True,
    )
    print(f"\n[+] Đã hoàn thành suy luận trên {len(results)} ảnh! Kết quả lưu tại: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()