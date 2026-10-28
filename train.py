"""
train.py - CLI entrypoint for training TASDF models.
Thin wrapper delegating directly to the high-level TASDF facade API.
"""
import argparse
from tasdf import TASDF


def parse_args():
    parser = argparse.ArgumentParser(description="TASDF Semi-Supervised Training CLI")
    parser.add_argument("--config", type=str, default="configs/coco_blastocyst.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Model YAML or checkpoint .pt")
    parser.add_argument("--data", type=str, default="None", help="Dataset YAML path or directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Target compute device (cuda or cpu)")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP FP16")
    parser.add_argument("--project", type=str, default="runs/train", help="Run output directory")
    parser.add_argument("--name", type=str, default="exp", help="Run experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint")
    parser.add_argument("--info", action="store_true", help="Display model parameter statistics and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    model_src = args.model or args.config
    model = TASDF(model=model_src)

    if args.info:
        model.info(detailed=True)
        return

    # Nếu không truyền --data riêng, truyền trực tiếp đường dẫn file --config
    # để resolve_data_config định vị đúng thư mục gốc chứa file YAML
    data_src = args.data if (args.data and str(args.data).lower() != "none") else args.config

    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.batch is not None:
        kwargs["batch"] = args.batch
    if args.imgsz is not None:
        kwargs["imgsz"] = args.imgsz
    if args.lr is not None:
        kwargs["lr"] = args.lr
    if args.device is not None:
        kwargs["device"] = args.device

    results = model.train(
        data=data_src,
        project=args.project,
        name=args.name,
        resume=args.resume,
        amp=args.amp,
        **kwargs,
    )
    print(f"\n[+] Huấn luyện hoàn tất! Checkpoints lưu tại:")
    print(f"    - Best: {results['best_checkpoint']}")
    print(f"    - Last: {results['last_checkpoint']}")


if __name__ == "__main__":
    main()
