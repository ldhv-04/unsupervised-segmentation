#!/usr/bin/env python3
"""
scripts/prepare_folder_dataset.py
Công cụ chuẩn hóa bộ dữ liệu phôi nang từ:
  1. Thư mục chứa các file nhãn COCO (.json) + Thư mục chứa toàn bộ ảnh gốc
  2. Hoặc thư mục chứa ảnh mask rời (.png, .bmp, .tif)
Sang cấu trúc chuẩn của FolderEmbryoDataset:
  output_dir/
    ├── train/
    │   ├── images/ (100% ảnh train)
    │   └── masks/  (Chỉ chứa mask .png cho đúng tỷ lệ labeled_ratio, VD: 10%)
    ├── valid/
    │   ├── images/ (100% ảnh valid)
    │   └── masks/  (100% mask .png để đánh giá)
    └── test/ (nếu có)
"""
import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Chuẩn hóa dữ liệu COCO JSON / Mask PNG sang FolderEmbryoDataset")
    parser.add_argument("--raw_images", type=str, required=True, help="Thư mục chứa toàn bộ ảnh gốc")
    parser.add_argument(
        "--raw_masks",
        type=str,
        required=True,
        help="Thư mục chứa các file _annotations.coco.json (hoặc chứa các file ảnh mask .png)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục xuất dữ liệu đầu ra")
    parser.add_argument("--labeled_ratio", type=float, default=0.10, help="Tỷ lệ giữ nhãn tập train (mặc định: 0.10)")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Tỷ lệ tách tập valid nếu không chia sẵn theo JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định")
    return parser.parse_args()


def map_category_to_class_id(cat_name: str, cat_id: int) -> int:
    """Ánh xạ tên lớp hoặc ID trong COCO JSON về chuẩn [1: ZP, 2: TE, 3: ICM, 4: BC]."""
    name_upper = cat_name.upper()
    if "ZP" in name_upper or "ZONA" in name_upper:
        return 1
    if "TE" in name_upper or "TROPHECTODERM" in name_upper:
        return 2
    if "ICM" in name_upper or "INNER" in name_upper:
        return 3
    if any(k in name_upper for k in ("BC", "BLASTOCOEL", "CAVITY", "BLASTOCYST")):
        return 4

    # Fallback theo nhóm category_id của bộ dữ liệu Roboflow nếu tên bị khuyết
    if cat_id in (7, 8, 9):
        return 1
    if cat_id in (4, 5, 6):
        return 2
    if cat_id in (1, 2, 3):
        return 3
    return 0


def index_coco_jsons(mask_root: Path) -> dict[str, dict]:
    """
    Quét tất cả các file .json trong mask_root và lập chỉ mục theo file_name và stem.
    """
    json_files = sorted(mask_root.rglob("*.json"))
    coco_index: dict[str, dict] = {}

    if not json_files:
        return coco_index

    print(f"[+] Phát hiện {len(json_files)} tệp JSON trong '{mask_root}':")
    for jf in json_files:
        print(f"    - {jf.relative_to(mask_root)}")
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Nhận diện gợi ý split từ đường dẫn hoặc tên file JSON
        path_lower = str(jf.relative_to(mask_root)).lower()
        if "val" in path_lower:
            split_hint = "valid"
        elif "test" in path_lower:
            split_hint = "test"
        elif "train" in path_lower:
            split_hint = "train"
        else:
            split_hint = "unknown"

        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        img_id_to_anns: dict[int, list] = {}
        for ann in data.get("annotations", []):
            if ann.get("segmentation"):
                img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        for img_info in data.get("images", []):
            fname = img_info["file_name"]
            stem = Path(fname).stem
            anns = img_id_to_anns.get(img_info["id"], [])

            record = {
                "width": img_info.get("width"),
                "height": img_info.get("height"),
                "anns": anns,
                "cats": cats,
                "split_hint": split_hint,
            }
            # Lưu chỉ mục theo cả tên file đầy đủ, stem, và tiền tố trước '.rf.' (Roboflow)
            coco_index[fname] = record
            coco_index[stem] = record
            if ".rf." in stem:
                coco_index[stem.split(".rf.")[0]] = record

    return coco_index


def rasterize_coco_record(record: dict, target_size: tuple[int, int]) -> tuple[np.ndarray | None, set[int]]:
    """Vẽ mặt nạ đa lớp từ bản ghi COCO JSON."""
    anns = record.get("anns", [])
    if not anns:
        return None, set()

    target_w, target_h = target_size
    orig_w = record.get("width") or target_w
    orig_h = record.get("height") or target_h
    cats = record.get("cats", {})

    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    present_classes: set[int] = set()

    def draw_priority(ann):
        cname = cats.get(ann["category_id"], "")
        cid = map_category_to_class_id(cname, ann["category_id"])
        return {4: 0, 1: 1, 2: 2, 3: 3}.get(cid, 0)

    for ann in sorted(anns, key=draw_priority):
        cname = cats.get(ann["category_id"], "")
        class_id = map_category_to_class_id(cname, ann["category_id"])
        if class_id <= 0:
            continue
        for seg in ann.get("segmentation", []):
            if isinstance(seg, list) and len(seg) >= 6:
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], color=class_id)
                present_classes.add(class_id)

    if not present_classes:
        return None, set()

    if (orig_w, orig_h) != (target_w, target_h):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return mask, present_classes


def find_png_mask(mask_root: Path, stem: str, target_size: tuple[int, int]) -> tuple[np.ndarray | None, set[int]]:
    """Fallback đọc trực tiếp file ảnh mask (.png, .bmp, .tif) nếu không dùng JSON."""
    for ext in [".png", ".bmp", ".tif", ".jpg"]:
        for suffix in ["", "_mask", "_gt"]:
            cand = mask_root / f"{stem}{suffix}{ext}"
            if cand.exists():
                m_img = Image.open(cand).convert("L").resize(target_size, Image.NEAREST)
                m_arr = np.array(m_img, dtype=np.uint8)
                unique_vals = sorted(np.unique(m_arr).tolist())
                if max(unique_vals) > 4:
                    remapped = np.zeros_like(m_arr, dtype=np.uint8)
                    for idx, val in enumerate(unique_vals):
                        remapped[m_arr == val] = idx
                    m_arr = remapped
                classes = set(int(c) for c in np.unique(m_arr) if c > 0)
                return (m_arr if classes else None), classes
    return None, set()


def export_split(
    records: list[dict],
    out_split_dir: Path,
    keep_labeled_ids: set[str] | None = None,
) -> tuple[int, int]:
    """Lưu ảnh và mặt nạ ra thư mục split tương ứng."""
    img_out = out_split_dir / "images"
    msk_out = out_split_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    labeled_cnt = 0
    unlabeled_cnt = 0

    for item in records:
        img_p: Path = item["path"]
        stem = img_p.stem
        img = Image.open(img_p).convert("L")
        img.save(img_out / f"{stem}.png")

        should_save_mask = (keep_labeled_ids is None) or (stem in keep_labeled_ids)
        if should_save_mask and item["mask"] is not None:
            Image.fromarray(item["mask"]).save(msk_out / f"{stem}.png")
            labeled_cnt += 1
        else:
            unlabeled_cnt += 1

    return labeled_cnt, unlabeled_cnt


def main():
    args = parse_args()
    img_dir = Path(args.raw_images)
    mask_dir = Path(args.raw_masks)
    out_dir = Path(args.output_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục mask: {mask_dir}")

    coco_index = index_coco_jsons(mask_dir)

    img_files = sorted(
        [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif")]
    )
    if not img_files:
        raise RuntimeError(f"Không tìm thấy file ảnh nào trong: {img_dir}")

    print(f"[+] Đang đối sánh {len(img_files)} ảnh với dữ liệu nhãn...")

    all_items = []
    for img_p in img_files:
        with Image.open(img_p) as im:
            sz = im.size

        rec = (
            coco_index.get(img_p.name)
            or coco_index.get(img_p.stem)
            or (coco_index.get(img_p.stem.split(".rf.")[0]) if ".rf." in img_p.stem else None)
        )

        if rec is not None:
            mask_arr, classes = rasterize_coco_record(rec, sz)
            split_hint = rec["split_hint"]
        else:
            mask_arr, classes = find_png_mask(mask_dir, img_p.stem, sz)
            split_hint = "unknown"

        all_items.append({
            "path": img_p,
            "stem": img_p.stem,
            "mask": mask_arr,
            "classes": classes,
            "split_hint": split_hint,
        })

    total_matched = sum(1 for x in all_items if x["mask"] is not None)
    print(f"[+] Số ảnh tìm thấy nhãn hợp lệ: {total_matched}/{len(all_items)}")
    if total_matched == 0:
        raise RuntimeError(
            "Không khớp được bất kỳ nhãn nào giữa thư mục ảnh và thư mục mask! "
            "Hãy kiểm tra lại tên file_name trong các file .json có trùng với tên file trong thư mục image hay không."
        )

    rng = random.Random(args.seed)

    # Phân tách Train / Valid / Test
    has_explicit_splits = any(x["split_hint"] in ("train", "valid", "test") for x in all_items)
    explicit_val = [x for x in all_items if x["split_hint"] == "valid" and x["mask"] is not None]

    if has_explicit_splits and len(explicit_val) > 0:
        print("[+] Sử dụng phân vùng train/valid/test gốc từ các file JSON...")
        train_items = [x for x in all_items if x["split_hint"] in ("train", "unknown")]
        val_items = explicit_val
        test_items = [x for x in all_items if x["split_hint"] == "test" and x["mask"] is not None]
    else:
        print(f"[+] Tự động chia tập Valid ({args.val_ratio * 100:.0f}%) từ các ảnh có nhãn...")
        labeled_pool = [x for x in all_items if x["mask"] is not None]
        unlabeled_pool = [x for x in all_items if x["mask"] is None]
        rng.shuffle(labeled_pool)

        num_val = max(1, int(round(len(labeled_pool) * args.val_ratio)))
        val_items = labeled_pool[:num_val]
        train_items = labeled_pool[num_val:] + unlabeled_pool
        test_items = []

    # Chọn lọc phân tầng (Stratified Selection) đúng tỷ lệ labeled_ratio trên tập Train
    num_labeled_target = max(1, int(round(len(train_items) * args.labeled_ratio)))

    # Ưu tiên 1: Ảnh có đủ cả 3 cấu trúc ZP (1), TE (2), ICM (3)
    tier_full = [x["stem"] for x in train_items if x["mask"] is not None and {1, 2, 3}.issubset(x["classes"])]
    # Ưu tiên 2: Ảnh có ít nhất 1 cấu trúc có nhãn
    tier_partial = [
        x["stem"] for x in train_items
        if x["mask"] is not None and x["stem"] not in set(tier_full)
    ]

    tier_full.sort()
    tier_partial.sort()
    rng.shuffle(tier_full)
    rng.shuffle(tier_partial)

    selected_train_stems = set((tier_full + tier_partial)[:num_labeled_target])

    # Xuất dữ liệu ra đĩa
    tr_l, tr_u = export_split(train_items, out_dir / "train", keep_labeled_ids=selected_train_stems)
    va_l, _ = export_split(val_items, out_dir / "valid", keep_labeled_ids=None)
    te_l = 0
    if test_items:
        te_l, _ = export_split(test_items, out_dir / "test", keep_labeled_ids=None)

    print("=" * 72)
    print(f"HOÀN TẤT CHUẨN HÓA DỮ LIỆU TẠI: {out_dir.resolve()}")
    print(f" - Train có nhãn (Labeled):      {tr_l} ảnh (Đủ cấu trúc ưu tiên)")
    print(f" - Train không nhãn (Unlabeled): {tr_u} ảnh")
    print(f" - Valid (100% Labeled):         {va_l} ảnh")
    if test_items:
        print(f" - Test (100% Labeled):          {te_l} ảnh")
    print("=" * 72)


if __name__ == "__main__":
    main()