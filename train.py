import os
from pathlib import Path
import gc
import json
import shutil as shutil_lib
import shutil
import subprocess
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 512
BATCH = 8
WORKERS = 4

SPLIT_SEED = 42
VAL_SIZE = 3000
CSV_SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

QUICK_CHECK_MODE = os.environ.get("YOLO_QUICK_CHECK", "0") == "1"
QUICK_CHECK_JOB_LIMIT = int(os.environ.get("YOLO_QUICK_CHECK_JOB_LIMIT", "1"))
QUICK_CHECK_TRAIN_SAMPLES = int(os.environ.get("YOLO_QUICK_CHECK_TRAIN_SAMPLES", "128"))
QUICK_CHECK_VAL_SAMPLES = int(os.environ.get("YOLO_QUICK_CHECK_VAL_SAMPLES", "32"))
QUICK_CHECK_TEST_SAMPLES = int(os.environ.get("YOLO_QUICK_CHECK_TEST_SAMPLES", "32"))
QUICK_CHECK_EPOCHS = int(os.environ.get("YOLO_QUICK_CHECK_EPOCHS", "1"))
QUICK_CHECK_IMGSZ = int(os.environ.get("YOLO_QUICK_CHECK_IMGSZ", "640"))
QUICK_CHECK_BATCH = int(os.environ.get("YOLO_QUICK_CHECK_BATCH", "4"))
FORCE_RECOPY = os.environ.get("YOLO_FORCE_RECOPY", "0") == "1"
FORCE_REUNPACK = os.environ.get("YOLO_FORCE_REUNPACK", "0") == "1"
KEEP_STAGE = os.environ.get("YOLO_KEEP_STAGE", "0") == "1"
CLASS_MODE = os.environ.get("YOLO_CLASS_MODE", "22").strip()
DEFAULT_LABEL_CSV = Path(__file__).with_name("annotations_all_merged_other_1000nf.csv")
LABEL_CSV = os.environ.get(
    "YOLO_LABEL_CSV",
    str(DEFAULT_LABEL_CSV) if DEFAULT_LABEL_CSV.exists() else "",
).strip()
CSV_BBOX_FORMAT = os.environ.get("YOLO_CSV_BBOX_FORMAT", "auto").strip().lower()
CSV_BACKGROUND_CLASS = os.environ.get("YOLO_CSV_BACKGROUND_CLASS", "No finding").strip()
CSV_IMAGE_ID_COL = os.environ.get("YOLO_CSV_IMAGE_ID_COL", "image_id").strip()
CSV_CLASS_COL = os.environ.get("YOLO_CSV_CLASS_COL", "class_name").strip()
CSV_SAMPLE_ID_COL = os.environ.get("YOLO_CSV_SAMPLE_ID_COL", "sample_id").strip()
CSV_SOURCE_COL = os.environ.get("YOLO_CSV_SOURCE_COL", "source").strip()
CSV_FILENAME_COL = os.environ.get("YOLO_CSV_FILENAME_COL", "").strip()
CSV_SCORE_COL = os.environ.get("YOLO_CSV_SCORE_COL", "score").strip()
CSV_IMAGE_WIDTH_COL = os.environ.get("YOLO_CSV_IMAGE_WIDTH_COL", "").strip()
CSV_IMAGE_HEIGHT_COL = os.environ.get("YOLO_CSV_IMAGE_HEIGHT_COL", "").strip()
WBF_IOU_THRESHOLD = float(os.environ.get("YOLO_WBF_IOU_THRESHOLD", "0.5"))
WBF_SKIP_BOX_THRESHOLD = float(os.environ.get("YOLO_WBF_SKIP_BOX_THRESHOLD", "0.0"))
CSV_RUNTIME_REBUILD = os.environ.get("YOLO_CSV_RUNTIME_REBUILD", "1") == "1"

OFFICIAL_14_CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]
DROP_8_CLASS_NAMES = [
    "Clavicle fracture",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Rib fracture",
]

# remote gốc trên rclone
REMOTE_BASE = os.environ.get("RCLONE_REMOTE_BASE", "rclone:")

# mỗi preprocessing là 1 folder dataset trên drive
PREPROCESS_JOBS = [
    {"name": "clahe", "remote_subdir": "vinbigdata-CLAHE-png"},
    {"name": "histogram_eq", "remote_subdir": "vinbigdata-HistogramEq-png"},
    {"name": "adaptive_preprocessed", "remote_subdir": "vinbigdata-adaptive-preprocessed"},
    {"name": "percentile", "remote_subdir": "vinbigdata-percentile-png"},
    {"name": "raw_minmax", "remote_subdir": "vinbigdata-raw_minmax-png"},
    {"name": "rescale_minmax", "remote_subdir": "vinbigdata-rescale_minmax-png"},
    {"name": "base_dataset", "remote_subdir": "vindr_yolo_dataset"},
    {"name": "LUT", "remote_subdir": "vinbigdata-lut-png"},
    {"name": "expert_window", "remote_subdir": "vinbigdata-expert-png"},
]

# local workspace
WORK_ROOT = Path(os.environ.get("YOLO_WORK_ROOT", "./yolo_preprocess_runner")).expanduser().resolve()
DOWNLOAD_ROOT = WORK_ROOT / "downloads"
EXTRACT_ROOT = WORK_ROOT / "extracted"
RUNS_ROOT = WORK_ROOT / "runs"
RESULTS_ROOT = WORK_ROOT / "results"
SUMMARY_CSV = WORK_ROOT / "training_summary.csv"
MPLCONFIGDIR = WORK_ROOT / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

for d in [WORK_ROOT, DOWNLOAD_ROOT, EXTRACT_ROOT, RUNS_ROOT, RESULTS_ROOT, MPLCONFIGDIR]:
    d.mkdir(parents=True, exist_ok=True)

print("DEVICE:", DEVICE)
print("WORK_ROOT:", WORK_ROOT)
print("QUICK_CHECK_MODE:", QUICK_CHECK_MODE)
print("FORCE_RECOPY:", FORCE_RECOPY)
print("FORCE_REUNPACK:", FORCE_REUNPACK)
print("KEEP_STAGE:", KEEP_STAGE)
print("CLASS_MODE:", CLASS_MODE)
print("LABEL_CSV:", LABEL_CSV if LABEL_CSV else "(disabled)")


# =========================
# HELPERS
# =========================
def run_cmd(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def ensure_command_exists(command_name: str):
    if shutil_lib.which(command_name) is None:
        raise RuntimeError(
            f"Thiếu lệnh '{command_name}'. Cài trước rồi chạy lại. "
            f"Ví dụ với rclone: curl https://rclone.org/install.sh | sudo bash"
        )


def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def safe_remove(path: Path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=True)


def clear_local_stage():
    safe_rmtree(DOWNLOAD_ROOT)
    safe_rmtree(EXTRACT_ROOT)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)


def rclone_copy(remote_path: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "rclone", "copy",
        remote_path,
        str(local_path),
        "--transfers", "4",
        "--checkers", "4",
        "--stats", "10s",
        "--retries", "3",
    ])


def resolve_remote_path(job):
    if job.get("remote_path"):
        return job["remote_path"]
    remote_subdir = job["remote_subdir"]
    return f"{REMOTE_BASE}{remote_subdir}"


def unpack_archives_in_dir(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    archive_suffixes = {".zip", ".tar", ".gz", ".tgz"}
    archives = []
    for p in sorted(src_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in archive_suffixes:
            archives.append(p)

    if not archives:
        return False

    for arc in archives:
        print("Unpacking:", arc)
        if arc.suffix.lower() == ".zip":
            with zipfile.ZipFile(arc, "r") as zf:
                zf.extractall(dst_dir)
        else:
            shutil.unpack_archive(str(arc), str(dst_dir))

    sync_non_archive_items(src_dir, dst_dir)

    return True


def find_dataset_root(search_root: Path):
    if not search_root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset root trong {search_root}")

    if (search_root / "data.yaml").exists():
        return search_root
    if (search_root / "images").exists() and (search_root / "labels").exists():
        return search_root
    if (search_root / "images").exists() and (search_root / "annotations").exists():
        return search_root

    for p in sorted(search_root.rglob("data.yaml")):
        return p.parent

    for p in sorted(search_root.rglob("*")):
        if not p.is_dir():
            continue
        if (p / "images").exists() and (p / "labels").exists():
            return p
        if (p / "images").exists() and (p / "annotations").exists():
            return p

    raise FileNotFoundError(f"Không tìm thấy dataset root trong {search_root}")


def try_find_dataset_root(search_root: Path):
    if not search_root.exists():
        return None
    try:
        return find_dataset_root(search_root)
    except FileNotFoundError:
        return None


def dir_has_any_files(path: Path):
    return path.exists() and any(path.iterdir())


def sync_non_archive_items(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return
    archive_suffixes = {".zip", ".tar", ".gz", ".tgz"}
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in archive_suffixes:
            continue
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def repair_extracted_sidecars(dataset_root: Path, local_download_dir: Path):
    annotations_dir = dataset_root / "annotations"
    needs_annotations = (
        not annotations_dir.exists()
        or not (annotations_dir / "instances_train.json").exists()
        or not (annotations_dir / "instances_val.json").exists()
    )
    if needs_annotations and dir_has_any_files(local_download_dir):
        print("Repair extracted sidecars from downloads:", local_download_dir)
        sync_non_archive_items(local_download_dir, dataset_root)


def list_archives(path: Path):
    if not path.exists():
        return []
    archive_suffixes = {".zip", ".tar", ".gz", ".tgz"}
    return [
        p for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix.lower() in archive_suffixes
    ]


def has_archives(path: Path):
    return len(list_archives(path)) > 0


def reset_job_stage(local_download_dir: Path, local_extract_dir: Path):
    safe_rmtree(local_download_dir)
    safe_rmtree(local_extract_dir)
    local_download_dir.mkdir(parents=True, exist_ok=True)
    local_extract_dir.mkdir(parents=True, exist_ok=True)


def ensure_local_job_stage(remote_path: str, local_download_dir: Path, local_extract_dir: Path):
    local_download_dir.mkdir(parents=True, exist_ok=True)
    local_extract_dir.mkdir(parents=True, exist_ok=True)

    if FORCE_RECOPY:
        print("YOLO_FORCE_RECOPY=1 -> reset local stage")
        reset_job_stage(local_download_dir, local_extract_dir)

    if not FORCE_REUNPACK:
        dataset_root = try_find_dataset_root(local_extract_dir)
        if dataset_root is not None:
            print("Reuse extracted dataset:", dataset_root)
            return dataset_root

    if dir_has_any_files(local_extract_dir):
        print("Found extracted files:", local_extract_dir)
        if dir_has_any_files(local_download_dir):
            print("Sync non-archive files from downloads:", local_download_dir)
            sync_non_archive_items(local_download_dir, local_extract_dir)

        dataset_root = try_find_dataset_root(local_extract_dir)
        if dataset_root is not None:
            print("Reuse extracted dataset after sync:", dataset_root)
            return dataset_root

        if not has_archives(local_download_dir):
            print("Extracted tree is incomplete and downloads has no archives. Refresh local stage.")
            reset_job_stage(local_download_dir, local_extract_dir)

    dataset_root = try_find_dataset_root(local_download_dir)
    if dataset_root is not None and not FORCE_REUNPACK:
        print("Reuse dataset directly from downloads:", dataset_root)
        return dataset_root

    if not dir_has_any_files(local_download_dir):
        print("Local downloads empty -> copy from remote")
        rclone_copy(remote_path, local_download_dir)
    elif not has_archives(local_download_dir) and try_find_dataset_root(local_download_dir) is None:
        print("Downloaded files are incomplete and not directly usable. Re-copy from remote.")
        reset_job_stage(local_download_dir, local_extract_dir)
        rclone_copy(remote_path, local_download_dir)

    dataset_root = try_find_dataset_root(local_download_dir)
    if dataset_root is not None and not FORCE_REUNPACK:
        print("Dataset root found in downloads:", dataset_root)
        return dataset_root

    if has_archives(local_download_dir):
        print("Archives detected in downloads -> unpack")
        unpack_archives_in_dir(local_download_dir, local_extract_dir)
    else:
        print("No archives detected in downloads")
        sync_non_archive_items(local_download_dir, local_extract_dir)

    dataset_root = try_find_dataset_root(local_extract_dir)
    if dataset_root is not None:
        print("Dataset root found in extracted:", dataset_root)
        return dataset_root

    dataset_root = try_find_dataset_root(local_download_dir)
    if dataset_root is not None:
        print("Dataset root found in downloads after refresh:", dataset_root)
        return dataset_root

    raise FileNotFoundError(
        f"Không tìm thấy dataset root sau khi đã thử copy/unpack."
        f" downloads={local_download_dir} extracted={local_extract_dir}"
    )


def list_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    if not folder.exists():
        return files
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def maybe_limit_files(files, limit: int):
    if limit <= 0 or len(files) <= limit:
        return files
    return files[:limit]


def validate_class_mode():
    if CLASS_MODE not in {"22", "14"}:
        raise ValueError(
            f"YOLO_CLASS_MODE phải là '22' hoặc '14', nhưng đang là: {CLASS_MODE}"
        )


def sanitize_class_name(name: str):
    return str(name).strip()


def normalize_names_field(names):
    if isinstance(names, dict):
        return [name for _, name in sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])]
    if isinstance(names, list):
        return list(names)
    raise ValueError(f"names field không hợp lệ: {type(names)}")


def build_names_dict(class_names):
    return {i: name for i, name in enumerate(class_names)}


def should_use_14class_mode():
    return CLASS_MODE == "14"


def ensure_runtime_symlink(src: Path, dst: Path):
    safe_remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)


def get_14class_idx_map_from_names(class_names):
    keep_set = set(OFFICIAL_14_CLASS_NAMES)
    target_name_to_idx = {name: idx for idx, name in enumerate(OFFICIAL_14_CLASS_NAMES)}
    return {
        src_idx: target_name_to_idx[name]
        for src_idx, name in enumerate(class_names)
        if name in keep_set
    }


def filter_yolo_label_file(src: Path, dst: Path, idx_map: dict):
    kept_lines = []
    if src.exists():
        for raw_line in src.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_idx = int(float(parts[0]))
            if cls_idx not in idx_map:
                continue
            parts[0] = str(idx_map[cls_idx])
            kept_lines.append(" ".join(parts))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))


def build_yolo_14class_dataset(source_root: Path, source_yaml: Path):
    with open(source_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    class_names = normalize_names_field(data_cfg["names"])
    idx_map = get_14class_idx_map_from_names(class_names)
    runtime_root = source_root / ".yolo_runtime_14class"
    safe_remove(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    ensure_runtime_symlink(source_root / "images", runtime_root / "images")

    src_labels_root = source_root / "labels"
    dst_labels_root = runtime_root / "labels"
    for src_label in sorted(src_labels_root.rglob("*.txt")):
        rel = src_label.relative_to(src_labels_root)
        filter_yolo_label_file(src_label, dst_labels_root / rel, idx_map)

    runtime_yaml = runtime_root / "data.yaml"
    runtime_cfg = {
        "path": str(runtime_root),
        "train": "images/train",
        "val": "images/val",
        "names": build_names_dict(OFFICIAL_14_CLASS_NAMES),
    }
    with open(runtime_yaml, "w") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    print("Built 14-class YOLO runtime dataset:", runtime_root)
    print("Dropped classes:", DROP_8_CLASS_NAMES)
    return runtime_root, runtime_yaml


def coco_bbox_to_yolo_line(bbox, img_w, img_h, cls_idx: int):
    x, y, w, h = bbox
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None

    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    width = w / img_w
    height = h / img_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    if width <= 0.0 or height <= 0.0:
        return None

    return f"{cls_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def collect_coco_categories(annotation_paths):
    categories = {}
    for ann_path in annotation_paths:
        if not ann_path.exists():
            continue
        with open(ann_path, "r") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            categories[int(cat["id"])] = sanitize_class_name(cat["name"])
    if not categories:
        raise ValueError("Không tìm thấy categories trong COCO annotations")
    sorted_items = sorted(categories.items(), key=lambda x: x[0])
    class_names = [name for _, name in sorted_items]
    cat_id_to_class_idx = {cat_id: idx for idx, (cat_id, _) in enumerate(sorted_items)}
    return class_names, cat_id_to_class_idx


def collect_coco_categories_14class(annotation_paths):
    categories = {}
    keep_set = set(OFFICIAL_14_CLASS_NAMES)
    for ann_path in annotation_paths:
        if not ann_path.exists():
            continue
        with open(ann_path, "r") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            name = sanitize_class_name(cat["name"])
            if name in keep_set:
                categories[int(cat["id"])] = name
    if not categories:
        raise ValueError("Không tìm thấy categories 14-class trong COCO annotations")
    target_name_to_idx = {name: idx for idx, name in enumerate(OFFICIAL_14_CLASS_NAMES)}
    cat_id_to_class_idx = {
        cat_id: target_name_to_idx[name]
        for cat_id, name in categories.items()
    }
    return list(OFFICIAL_14_CLASS_NAMES), cat_id_to_class_idx


def convert_coco_split_to_yolo(annotation_path: Path, image_dir: Path, label_dir: Path, cat_id_to_class_idx: dict):
    if not annotation_path.exists():
        return {"images": 0, "labels": 0, "annotations": 0}

    label_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, "r") as f:
        data = json.load(f)

    images = {int(img["id"]): img for img in data.get("images", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image_id = int(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(ann)

    label_count = 0
    ann_count = 0

    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        img_w = float(image_info["width"])
        img_h = float(image_info["height"])
        image_path = resolve_coco_image_path(image_dir, file_name)
        if image_path is None:
            raise FileNotFoundError(f"Thiếu ảnh {file_name} trong {image_dir}")
        label_path = label_dir / f"{image_path.stem}.txt"

        lines = []
        for ann in anns_by_image.get(image_id, []):
            category_id = int(ann["category_id"])
            if category_id not in cat_id_to_class_idx:
                continue
            line = coco_bbox_to_yolo_line(
                bbox=ann["bbox"],
                img_w=img_w,
                img_h=img_h,
                cls_idx=cat_id_to_class_idx[category_id],
            )
            if line is not None:
                lines.append(line)
                ann_count += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        label_count += 1

    return {
        "images": len(images),
        "labels": label_count,
        "annotations": ann_count,
    }


def resolve_coco_image_path(image_dir: Path, file_name: str):
    file_path = Path(file_name)

    # case 1: COCO stores bare filename
    candidate = image_dir / file_path
    if candidate.exists():
        return candidate

    # case 2: COCO stores path like images/train/xxx.png or ./images/train/xxx.png
    parts = [part for part in file_path.parts if part not in {"."}]
    for idx, part in enumerate(parts):
        if part == "images" and idx + 1 < len(parts):
            trimmed = Path(*parts[idx + 2:]) if idx + 2 < len(parts) else Path(parts[-1])
            candidate = image_dir / trimmed
            if candidate.exists():
                return candidate

    # case 3: fallback by basename
    matches = list(image_dir.rglob(file_path.name))
    if matches:
        return matches[0]

    return None


def ensure_yolo_dataset(dataset_root: Path):
    data_yaml = dataset_root / "data.yaml"
    labels_dir = dataset_root / "labels"
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if should_use_14class_mode():
        if data_yaml.exists() and labels_dir.exists():
            return build_yolo_14class_dataset(dataset_root, data_yaml)

        if not images_dir.exists() or not annotations_dir.exists():
            raise FileNotFoundError(
                f"Dataset ở {dataset_root} không có đủ data.yaml/labels hoặc images/annotations để convert"
            )

        train_ann = annotations_dir / "instances_train.json"
        val_ann = annotations_dir / "instances_val.json"
        if not train_ann.exists() or not val_ann.exists():
            raise FileNotFoundError(
                f"Thiếu COCO annotations trong {annotations_dir}. Cần instances_train.json và instances_val.json"
            )

        class_names, cat_id_to_class_idx = collect_coco_categories_14class([train_ann, val_ann])
        runtime_root = dataset_root / ".yolo_runtime_14class"
        safe_remove(runtime_root)
        runtime_root.mkdir(parents=True, exist_ok=True)
        ensure_runtime_symlink(images_dir, runtime_root / "images")
        runtime_labels_dir = runtime_root / "labels"
        (runtime_labels_dir / "train").mkdir(parents=True, exist_ok=True)
        (runtime_labels_dir / "val").mkdir(parents=True, exist_ok=True)

        print("Converting COCO dataset to 14-class YOLO format:", dataset_root)
        print("Kept classes:", class_names)

        train_stats = convert_coco_split_to_yolo(
            annotation_path=train_ann,
            image_dir=runtime_root / "images" / "train",
            label_dir=runtime_labels_dir / "train",
            cat_id_to_class_idx=cat_id_to_class_idx,
        )
        val_stats = convert_coco_split_to_yolo(
            annotation_path=val_ann,
            image_dir=runtime_root / "images" / "val",
            label_dir=runtime_labels_dir / "val",
            cat_id_to_class_idx=cat_id_to_class_idx,
        )

        runtime_yaml = runtime_root / "data.yaml"
        data_cfg = {
            "path": str(runtime_root),
            "train": "images/train",
            "val": "images/val",
            "names": build_names_dict(class_names),
        }
        with open(runtime_yaml, "w") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)

        print("COCO -> 14-class YOLO train stats:", train_stats)
        print("COCO -> 14-class YOLO val stats:", val_stats)
        print("Created:", runtime_yaml)
        return runtime_root, runtime_yaml

    if data_yaml.exists() and labels_dir.exists():
        return dataset_root, data_yaml

    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(
            f"Dataset ở {dataset_root} không có đủ data.yaml/labels hoặc images/annotations để convert"
        )

    train_ann = annotations_dir / "instances_train.json"
    val_ann = annotations_dir / "instances_val.json"
    if not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError(
            f"Thiếu COCO annotations trong {annotations_dir}. Cần instances_train.json và instances_val.json"
        )

    class_names, cat_id_to_class_idx = collect_coco_categories([train_ann, val_ann])
    print("Converting COCO dataset to YOLO format:", dataset_root)
    print("Detected classes:", class_names)

    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)

    train_stats = convert_coco_split_to_yolo(
        annotation_path=train_ann,
        image_dir=images_dir / "train",
        label_dir=labels_dir / "train",
        cat_id_to_class_idx=cat_id_to_class_idx,
    )
    val_stats = convert_coco_split_to_yolo(
        annotation_path=val_ann,
        image_dir=images_dir / "val",
        label_dir=labels_dir / "val",
        cat_id_to_class_idx=cat_id_to_class_idx,
    )

    data_cfg = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(data_yaml, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    print("COCO -> YOLO train stats:", train_stats)
    print("COCO -> YOLO val stats:", val_stats)
    print("Created:", data_yaml)
    return dataset_root, data_yaml


def get_runtime_train_params():
    if QUICK_CHECK_MODE:
        return {
            "epochs": QUICK_CHECK_EPOCHS,
            "imgsz": QUICK_CHECK_IMGSZ,
            "batch": min(BATCH, QUICK_CHECK_BATCH),
        }
    return {
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
    }


def prepare_runtime_yaml(dataset_root: Path, original_yaml: Path | None = None):
    if original_yaml is None:
        original_yaml = dataset_root / "data.yaml"
    if not original_yaml.exists():
        raise FileNotFoundError(f"Thiếu data.yaml trong {dataset_root}")

    with open(original_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    train_pool_dir = dataset_root / "images" / "train"
    test_dir = dataset_root / "images" / "val"  # val cũ trong export được coi là test thật

    train_pool_files = list_image_files(train_pool_dir)
    if not train_pool_files:
        raise FileNotFoundError(f"Không có ảnh trong {train_pool_dir}")

    rng = np.random.default_rng(SPLIT_SEED)
    val_size = min(VAL_SIZE, len(train_pool_files))
    val_indices = set(rng.choice(len(train_pool_files), size=val_size, replace=False).tolist())

    # Keep symlink-based paths intact for 14-class runtime datasets.
    # Using resolve() would collapse images back to the source dataset root,
    # causing Ultralytics to look for labels in the wrong labels/ directory.
    runtime_train_files = [
        str(p.absolute()) for i, p in enumerate(train_pool_files) if i not in val_indices
    ]
    runtime_val_files = [
        str(p.absolute()) for i, p in enumerate(train_pool_files) if i in val_indices
    ]
    runtime_test_files = [
        str(p.absolute()) for p in list_image_files(test_dir)
    ] if test_dir.exists() else []

    if QUICK_CHECK_MODE:
        runtime_train_files = maybe_limit_files(runtime_train_files, QUICK_CHECK_TRAIN_SAMPLES)
        runtime_val_files = maybe_limit_files(runtime_val_files, QUICK_CHECK_VAL_SAMPLES)
        runtime_test_files = maybe_limit_files(runtime_test_files, QUICK_CHECK_TEST_SAMPLES)

    train_txt = dataset_root / "train.runtime.txt"
    val_txt = dataset_root / "val.runtime.txt"
    test_txt = dataset_root / "test.runtime.txt"

    train_txt.write_text("\n".join(runtime_train_files) + "\n")
    val_txt.write_text("\n".join(runtime_val_files) + "\n")
    if runtime_test_files:
        test_txt.write_text("\n".join(runtime_test_files) + "\n")

    runtime_cfg = {
        "path": str(dataset_root),
        "train": str(train_txt),
        "val": str(val_txt),
        "names": data_cfg["names"],
    }
    if runtime_test_files:
        runtime_cfg["test"] = str(test_txt)

    runtime_yaml = dataset_root / "data.runtime.yaml"
    with open(runtime_yaml, "w") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    split_info = {
        "train_pool_images": len(train_pool_files),
        "train_images": len(runtime_train_files),
        "val_images": len(runtime_val_files),
        "test_images": len(runtime_test_files),
        "train_txt": str(train_txt),
        "val_txt": str(val_txt),
        "test_txt": str(test_txt) if runtime_test_files else "",
    }

    return runtime_yaml, runtime_cfg, split_info


def csv_mode_enabled():
    return bool(LABEL_CSV)


def normalize_split_ratios(split_ratios):
    split_ratios = dict(split_ratios)
    total = sum(float(v) for v in split_ratios.values())
    if total <= 0:
        raise ValueError("CSV split ratios must sum to > 0")
    return {k: float(v) / total for k, v in split_ratios.items()}


def compute_target_sizes(n_items, split_ratios):
    raw = {split: n_items * ratio for split, ratio in split_ratios.items()}
    sizes = {split: int(np.floor(value)) for split, value in raw.items()}
    remaining = n_items - sum(sizes.values())

    order = sorted(
        split_ratios.keys(),
        key=lambda split: (raw[split] - sizes[split], split),
        reverse=True,
    )
    for split in order[:remaining]:
        sizes[split] += 1

    nonzero_splits = [split for split, ratio in split_ratios.items() if ratio > 0]
    if n_items >= len(nonzero_splits):
        for split in nonzero_splits:
            if sizes[split] == 0:
                donor = max(sizes, key=sizes.get)
                sizes[donor] -= 1
                sizes[split] += 1

    return sizes


def is_background_class(class_name):
    normalized = str(class_name).strip().lower().replace("_", " ")
    background_names = {
        CSV_BACKGROUND_CLASS.strip().lower().replace("_", " "),
        "no finding",
        "background",
        "negative",
    }
    return normalized in background_names


def first_existing_column(df, explicit_col, candidates):
    if explicit_col and explicit_col in df.columns:
        return explicit_col
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_label_csv(csv_path: Path):
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Không thấy YOLO_LABEL_CSV: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    print("CSV label path:", csv_path)
    print("CSV shape:", df_raw.shape)
    print("CSV columns:", df_raw.columns.tolist())

    required = [CSV_IMAGE_ID_COL, CSV_CLASS_COL, "x_min", "y_min", "x_max", "y_max"]
    missing = [col for col in required if col not in df_raw.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {missing}")

    df = df_raw.copy()
    df["image_id"] = df[CSV_IMAGE_ID_COL].astype("string").str.strip()
    df["class_name"] = df[CSV_CLASS_COL].astype("string").str.strip()
    df = df.dropna(subset=["image_id", "class_name"]).copy()

    source_col = first_existing_column(
        df,
        CSV_SOURCE_COL,
        ["source", "original_split", "split"],
    )
    if source_col:
        df["source"] = df[source_col].astype("string").str.strip()
        print("CSV source column:", source_col)
    else:
        df["source"] = ""
        print("CSV source column: not found")

    if CSV_SAMPLE_ID_COL and CSV_SAMPLE_ID_COL in df.columns:
        df["sample_id"] = df[CSV_SAMPLE_ID_COL].astype("string").str.strip()
    else:
        df["sample_id"] = df["image_id"].astype(str)

    filename_col = first_existing_column(
        df,
        CSV_FILENAME_COL,
        ["file_name", "filename", "image_path", "path"],
    )
    if filename_col:
        df["csv_file_name"] = df[filename_col].astype("string").str.strip()
    else:
        df["csv_file_name"] = ""

    width_col = first_existing_column(
        df,
        CSV_IMAGE_WIDTH_COL,
        ["image_width", "img_width", "original_width", "orig_width", "width"],
    )
    height_col = first_existing_column(
        df,
        CSV_IMAGE_HEIGHT_COL,
        ["image_height", "img_height", "original_height", "orig_height", "height"],
    )
    if width_col and height_col:
        df["source_width"] = pd.to_numeric(df[width_col], errors="coerce")
        df["source_height"] = pd.to_numeric(df[height_col], errors="coerce")
        print("CSV image size columns:", width_col, height_col)
    else:
        df["source_width"] = np.nan
        df["source_height"] = np.nan
        print("CSV image size columns: not found; pixel boxes will use actual image size")

    if CSV_SCORE_COL and CSV_SCORE_COL in df.columns:
        df["box_score"] = pd.to_numeric(df[CSV_SCORE_COL], errors="coerce").fillna(1.0)
        print("CSV WBF score column:", CSV_SCORE_COL)
    else:
        df["box_score"] = 1.0
        print("CSV WBF score column: not found; all boxes weight=1")

    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_background"] = df["class_name"].map(is_background_class)
    all_sample_ids = sorted(df["sample_id"].dropna().astype(str).str.strip().unique().tolist())

    ann_df = df[~df["is_background"]].copy()
    ann_df = ann_df.dropna(subset=["sample_id", "class_name", "x_min", "y_min", "x_max", "y_max"]).copy()
    ann_df = ann_df[
        (ann_df["x_max"] > ann_df["x_min"])
        & (ann_df["y_max"] > ann_df["y_min"])
        & (ann_df["box_score"] >= WBF_SKIP_BOX_THRESHOLD)
    ].copy()

    class_names = sorted(ann_df["class_name"].astype(str).unique().tolist())
    if not class_names:
        raise ValueError("CSV không có class bất thường hợp lệ sau khi bỏ background/No finding")

    print("CSV unique samples:", len(all_sample_ids))
    print("CSV abnormal samples:", ann_df["sample_id"].nunique())
    print("CSV background-only samples:", len(set(all_sample_ids) - set(ann_df["sample_id"].astype(str).unique())))
    print("CSV num classes:", len(class_names))
    print("CSV class names:", class_names)

    return df.reset_index(drop=True), ann_df.reset_index(drop=True), all_sample_ids, class_names


def compute_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def weighted_fuse_group(group_df):
    if len(group_df) <= 1:
        return group_df.copy()

    rows = []
    work = group_df.copy()
    work["_score_sort"] = pd.to_numeric(work["box_score"], errors="coerce").fillna(1.0)
    work = work.sort_values("_score_sort", ascending=False).reset_index(drop=True)

    clusters = []
    for idx, row in work.iterrows():
        box = row[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
        score = float(row.get("box_score", 1.0))
        best_cluster = None
        best_iou = 0.0

        for cluster_idx, cluster in enumerate(clusters):
            iou = compute_iou_xyxy(box, cluster["box"])
            if iou >= WBF_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_cluster = cluster_idx

        if best_cluster is None:
            clusters.append({
                "indices": [idx],
                "boxes": [box],
                "scores": [score],
                "box": box.copy(),
            })
            continue

        cluster = clusters[best_cluster]
        cluster["indices"].append(idx)
        cluster["boxes"].append(box)
        cluster["scores"].append(score)
        weights = np.asarray(cluster["scores"], dtype=float)
        weights = np.maximum(weights, 1e-6)
        cluster["box"] = np.average(np.asarray(cluster["boxes"], dtype=float), axis=0, weights=weights)

    for cluster in clusters:
        fused = work.loc[cluster["indices"]].iloc[0].copy()
        weights = np.asarray(cluster["scores"], dtype=float)
        weights = np.maximum(weights, 1e-6)
        fused_box = np.average(np.asarray(cluster["boxes"], dtype=float), axis=0, weights=weights)
        fused["x_min"] = float(fused_box[0])
        fused["y_min"] = float(fused_box[1])
        fused["x_max"] = float(fused_box[2])
        fused["y_max"] = float(fused_box[3])
        fused["box_score"] = float(np.mean(cluster["scores"]))
        fused["fusion_group_size"] = int(len(cluster["indices"]))
        fused["was_fused"] = bool(len(cluster["indices"]) > 1)
        rows.append(fused.drop(labels=["_score_sort"], errors="ignore"))

    return pd.DataFrame(rows)


def apply_wbf_to_annotations(ann_df):
    fused_groups = []
    grouped = ann_df.groupby(["sample_id", "class_name"], sort=False, group_keys=False)
    for _, group in grouped:
        fused_groups.append(weighted_fuse_group(group))

    if not fused_groups:
        return ann_df.copy()

    fused_df = pd.concat(fused_groups, ignore_index=True).reset_index(drop=True)
    print("WBF enabled")
    print("WBF IoU threshold:", WBF_IOU_THRESHOLD)
    print("Boxes before WBF:", len(ann_df))
    print("Boxes after WBF:", len(fused_df))
    print("Rows created from fused clusters:", int(fused_df.get("was_fused", pd.Series(dtype=bool)).sum()))
    return fused_df


def build_image_class_sets(ann_df, sample_ids):
    sample_ids = sorted(set(map(str, sample_ids)))
    grouped = (
        ann_df.groupby("sample_id")["class_name"]
        .apply(lambda x: sorted(set(map(str, x))))
        .to_dict()
    )
    return {
        sample_id: grouped.get(sample_id, [CSV_BACKGROUND_CLASS])
        for sample_id in sample_ids
    }


def build_multilabel_stratified_split(ann_df, sample_ids, split_ratios=None, seed=42):
    split_ratios = normalize_split_ratios(split_ratios or CSV_SPLIT_RATIOS)
    rng = np.random.default_rng(seed)

    sample_ids = sorted(set(map(str, sample_ids)))
    image_class_sets = build_image_class_sets(ann_df, sample_ids)
    split_names = list(split_ratios.keys())

    target_sizes = compute_target_sizes(len(sample_ids), split_ratios)
    class_total = Counter(cls for classes in image_class_sets.values() for cls in classes)
    target_class_counts = {
        split: {cls: class_total[cls] * split_ratios[split] for cls in class_total}
        for split in split_names
    }

    current_sizes = {split: 0 for split in split_names}
    current_class_counts = {split: Counter() for split in split_names}
    split_sets = {split: set() for split in split_names}

    ordered_ids = sorted(
        sample_ids,
        key=lambda sample_id: (
            -sum(1.0 / max(class_total[cls], 1) for cls in image_class_sets[sample_id]),
            -len(image_class_sets[sample_id]),
            rng.random(),
        ),
    )

    for sample_id in ordered_ids:
        classes = image_class_sets[sample_id]
        candidates = [split for split in split_names if current_sizes[split] < target_sizes[split]]
        if not candidates:
            candidates = split_names

        best_split = None
        best_score = None

        for split in candidates:
            size_deficit = (target_sizes[split] - current_sizes[split]) / max(target_sizes[split], 1)
            class_deficit = 0.0
            over_penalty = 0.0

            for cls in classes:
                target = max(target_class_counts[split][cls], 1e-9)
                before = current_class_counts[split][cls]
                after = before + 1
                class_deficit += max(target - before, 0.0) / target
                over_penalty += max(after - target, 0.0) / target

            score = size_deficit + class_deficit - over_penalty + rng.random() * 1e-6
            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        split_sets[best_split].add(sample_id)
        current_sizes[best_split] += 1
        for cls in classes:
            current_class_counts[best_split][cls] += 1

    return split_sets, image_class_sets


def summarize_split_distribution(split_sets, image_class_sets):
    class_names = sorted({cls for classes in image_class_sets.values() for cls in classes})
    rows = []
    for cls in class_names:
        total = sum(1 for classes in image_class_sets.values() if cls in classes)
        row = {"class_name": cls, "total_images": total}
        for split, ids in split_sets.items():
            count = sum(1 for sample_id in ids if cls in image_class_sets[sample_id])
            row[f"{split}_images"] = count
            row[f"{split}_pct_of_class"] = round(100.0 * count / total, 2) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)


def image_source_aliases(source):
    source = str(source or "").strip().lower()
    if source == "train":
        return ["train"]
    if source == "test":
        return ["test", "val"]
    if source == "val":
        return ["val"]
    return []


def strip_source_prefix(value):
    value = str(value).strip()
    if "__" in value:
        return value.split("__", 1)[1]
    return value


def build_image_index(dataset_root: Path):
    images_root = dataset_root / "images"
    files = list_image_files(images_root)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy ảnh dưới {images_root}")

    by_stem = {}
    by_source_stem = {}
    for path in files:
        by_stem.setdefault(path.stem, []).append(path)
        rel_parts = path.relative_to(images_root).parts
        if len(rel_parts) >= 2:
            split_name = rel_parts[0].lower()
            by_source_stem.setdefault((split_name, path.stem), []).append(path)

    return by_stem, by_source_stem, files


def candidate_stems_for_row(row):
    values = []
    for value in [
        row.get("csv_file_name", ""),
        row.get("sample_id", ""),
        row.get("image_id", ""),
        strip_source_prefix(row.get("sample_id", "")),
    ]:
        if pd.isna(value):
            continue
        value = str(value).strip()
        if not value:
            continue
        values.append(Path(value).stem)
    return list(dict.fromkeys(values))


def resolve_image_for_row(row, by_stem, by_source_stem):
    stems = candidate_stems_for_row(row)
    source_aliases = image_source_aliases(row.get("source", ""))

    for alias in source_aliases:
        for stem in stems:
            matches = by_source_stem.get((alias, stem), [])
            if matches:
                return sorted(matches)[0]

    for stem in stems:
        matches = by_stem.get(stem, [])
        if matches:
            return sorted(matches)[0]

    return None


def build_image_records(dataset_root: Path, csv_df, sample_ids):
    by_stem, by_source_stem, all_image_files = build_image_index(dataset_root)
    lookup_df = (
        csv_df[csv_df["sample_id"].astype(str).isin(set(map(str, sample_ids)))]
        [["sample_id", "image_id", "source", "csv_file_name"]]
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )

    records = {}
    missing = []
    for row in lookup_df.itertuples(index=False):
        row_dict = row._asdict()
        path = resolve_image_for_row(row_dict, by_stem, by_source_stem)
        sample_id = str(row_dict["sample_id"])
        if path is None:
            missing.append(sample_id)
            continue
        records[sample_id] = {
            "sample_id": sample_id,
            "image_id": str(row_dict["image_id"]),
            "source": str(row_dict.get("source", "")),
            "path": path,
        }

    missing = sorted(set(missing) | (set(map(str, sample_ids)) - set(records.keys())))
    print("Dataset image files indexed:", len(all_image_files))
    print("CSV requested samples:", len(set(map(str, sample_ids))))
    print("CSV matched images:", len(records))
    print("CSV missing images:", len(missing))
    if missing:
        print("First missing CSV samples:", missing[:10])
    return records, missing


def collect_coco_image_size_lookup(dataset_root: Path):
    annotations_root = dataset_root / "annotations"
    if not annotations_root.exists():
        return {}

    size_lookup = {}
    for ann_path in sorted(annotations_root.rglob("*.json")):
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Skip unreadable annotation size file:", ann_path, e)
            continue

        for image_info in data.get("images", []):
            file_name = str(image_info.get("file_name", "")).strip()
            if not file_name:
                continue
            width = image_info.get("width")
            height = image_info.get("height")
            try:
                width = float(width)
                height = float(height)
            except Exception:
                continue
            if width <= 0 or height <= 0:
                continue

            stem = Path(file_name).stem
            size_lookup.setdefault(stem, (width, height))

    if size_lookup:
        print("Loaded image sizes from COCO annotations:", len(size_lookup))
    else:
        print("No image sizes loaded from COCO annotations")
    return size_lookup


def read_image_size(path: Path):
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass

    try:
        from PIL import Image
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception as e:
        raise RuntimeError(
            f"Không đọc được kích thước ảnh {path}. Cài opencv-python hoặc Pillow, "
            "hoặc thêm image_width/image_height vào CSV."
        ) from e


def sanitize_file_stem(value):
    text = str(value).strip()
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    return text


def symlink_image(src: Path, dst: Path):
    safe_remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.absolute())


def sample_source_size(rows):
    if rows is None or len(rows) == 0:
        return None
    widths = pd.to_numeric(rows.get("source_width", pd.Series(dtype=float)), errors="coerce").dropna()
    heights = pd.to_numeric(rows.get("source_height", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(widths) > 0 and len(heights) > 0 and float(widths.iloc[0]) > 0 and float(heights.iloc[0]) > 0:
        return float(widths.iloc[0]), float(heights.iloc[0])
    return None


def lookup_source_size(sample_id, rows, image_records, size_lookup):
    source_size = sample_source_size(rows)
    if source_size is not None:
        return source_size

    record = image_records.get(str(sample_id), {})
    candidates = [
        record.get("image_id", ""),
        strip_source_prefix(sample_id),
        Path(str(record.get("path", ""))).stem,
    ]
    for candidate in candidates:
        key = str(candidate).strip()
        if key in size_lookup:
            return size_lookup[key]
    return None


def detect_bbox_format(rows, source_size, actual_size):
    if CSV_BBOX_FORMAT != "auto":
        return CSV_BBOX_FORMAT
    if rows is None or len(rows) == 0:
        return "xyxy_pixel"
    coords = rows[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
    finite = coords[np.isfinite(coords)]
    if finite.size and np.nanmax(finite) <= 1.05 and np.nanmin(finite) >= -0.05:
        return "xyxy_norm"
    if source_size is not None:
        source_w, source_h = source_size
        max_x = np.nanmax(coords[:, [0, 2]]) if coords.size else 0.0
        max_y = np.nanmax(coords[:, [1, 3]]) if coords.size else 0.0
        if max_x > source_w * 1.25 or max_y > source_h * 1.25:
            raise ValueError(
                "CSV bbox lớn hơn nhiều so với source image size tìm được. "
                f"bbox_max=({max_x:.1f}, {max_y:.1f}), source_size=({source_w:.1f}, {source_h:.1f}). "
                "Với file annotations_all_merged_other_1000nf.csv, bbox đang là pixel ảnh gốc; "
                "hãy thêm cột image_width/image_height của ảnh gốc vào CSV, hoặc cung cấp annotation sidecar "
                "có width/height ảnh gốc."
            )
        return "xyxy_pixel"
    if source_size is None:
        actual_w, actual_h = actual_size
        max_x = np.nanmax(coords[:, [0, 2]]) if coords.size else 0.0
        max_y = np.nanmax(coords[:, [1, 3]]) if coords.size else 0.0
        if max_x > actual_w * 1.25 or max_y > actual_h * 1.25:
            raise ValueError(
                "CSV bbox có vẻ là pixel theo ảnh gốc nhưng CSV không có width/height. "
                "Thêm cột image_width/image_height hoặc set YOLO_CSV_IMAGE_WIDTH_COL/YOLO_CSV_IMAGE_HEIGHT_COL."
            )
    return "xyxy_pixel"


def row_to_yolo_line(row, class2id, source_size, actual_size, bbox_format):
    cls_idx = class2id[str(row["class_name"])]
    x1 = float(row["x_min"])
    y1 = float(row["y_min"])
    x2 = float(row["x_max"])
    y2 = float(row["y_max"])

    if bbox_format in {"xyxy_norm", "normalized_xyxy"}:
        x1 = float(np.clip(x1, 0.0, 1.0))
        y1 = float(np.clip(y1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))
        y2 = float(np.clip(y2, 0.0, 1.0))
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
    elif bbox_format in {"xyxy_pixel", "pixel_xyxy"}:
        denom_w, denom_h = source_size if source_size is not None else actual_size
        x1 = float(np.clip(x1, 0.0, denom_w))
        y1 = float(np.clip(y1, 0.0, denom_h))
        x2 = float(np.clip(x2, 0.0, denom_w))
        y2 = float(np.clip(y2, 0.0, denom_h))
        xc = ((x1 + x2) / 2.0) / denom_w
        yc = ((y1 + y2) / 2.0) / denom_h
        bw = (x2 - x1) / denom_w
        bh = (y2 - y1) / denom_h
    else:
        raise ValueError(
            f"YOLO_CSV_BBOX_FORMAT không hỗ trợ: {CSV_BBOX_FORMAT}. "
            "Dùng auto, xyxy_pixel hoặc xyxy_norm."
        )

    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    bw = float(np.clip(bw, 0.0, 1.0))
    bh = float(np.clip(bh, 0.0, 1.0))
    if bw <= 0.0 or bh <= 0.0:
        return None
    return f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def write_label_file(label_path: Path, rows, class2id, source_size, actual_size):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    bbox_format = detect_bbox_format(rows, source_size, actual_size)
    for _, row in rows.iterrows():
        line = row_to_yolo_line(row, class2id, source_size, actual_size, bbox_format)
        if line is not None:
            lines.append(line)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines), bbox_format


def maybe_limit_split_sets(split_sets):
    if not QUICK_CHECK_MODE:
        return split_sets
    limits = {
        "train": QUICK_CHECK_TRAIN_SAMPLES,
        "val": QUICK_CHECK_VAL_SAMPLES,
        "test": QUICK_CHECK_TEST_SAMPLES,
    }
    return {
        split: set(sorted(ids)[:limits.get(split, len(ids))])
        for split, ids in split_sets.items()
    }


def prepare_csv_runtime_dataset(dataset_root: Path, job_name: str):
    csv_df, ann_df, all_sample_ids, class_names = normalize_label_csv(Path(LABEL_CSV))
    fused_df = apply_wbf_to_annotations(ann_df)

    image_records, missing = build_image_records(dataset_root, csv_df, all_sample_ids)
    used_sample_ids = sorted(image_records.keys())
    if not used_sample_ids:
        raise ValueError(f"Không match được ảnh nào từ CSV với dataset {dataset_root}")

    fused_df = fused_df[fused_df["sample_id"].astype(str).isin(set(used_sample_ids))].copy()
    class_names = sorted(fused_df["class_name"].astype(str).unique().tolist())
    if not class_names:
        raise ValueError("Không còn bbox bất thường nào sau khi match ảnh từ CSV")

    class2id = {name: idx for idx, name in enumerate(class_names)}
    source_size_lookup = collect_coco_image_size_lookup(dataset_root)
    split_sets, image_class_sets = build_multilabel_stratified_split(
        ann_df=fused_df,
        sample_ids=used_sample_ids,
        split_ratios=CSV_SPLIT_RATIOS,
        seed=SPLIT_SEED,
    )
    split_sets = maybe_limit_split_sets(split_sets)

    runtime_root = dataset_root / ".csv_runtime"
    if CSV_RUNTIME_REBUILD:
        safe_remove(runtime_root)
    for split in ["train", "val", "test"]:
        (runtime_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (runtime_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    ann_by_sample = {
        str(sample_id): group.copy()
        for sample_id, group in fused_df.groupby("sample_id", sort=False)
    }

    split_counts = {}
    label_counts = {}
    bbox_format_counts = Counter()
    for split, ids in split_sets.items():
        image_count = 0
        label_count = 0
        for sample_id in sorted(ids):
            record = image_records[str(sample_id)]
            src_image = Path(record["path"])
            out_stem = sanitize_file_stem(sample_id)
            dst_image = runtime_root / "images" / split / f"{out_stem}{src_image.suffix.lower()}"
            dst_label = runtime_root / "labels" / split / f"{out_stem}.txt"
            symlink_image(src_image, dst_image)

            rows = ann_by_sample.get(str(sample_id), pd.DataFrame(columns=fused_df.columns))
            actual_size = read_image_size(src_image)
            source_size = lookup_source_size(
                sample_id=sample_id,
                rows=rows,
                image_records=image_records,
                size_lookup=source_size_lookup,
            )
            n_labels, bbox_format = write_label_file(
                label_path=dst_label,
                rows=rows,
                class2id=class2id,
                source_size=source_size,
                actual_size=actual_size,
            )
            bbox_format_counts[bbox_format] += 1
            image_count += 1
            label_count += n_labels

        split_counts[split] = image_count
        label_counts[split] = label_count

    runtime_yaml = runtime_root / "data.runtime.yaml"
    runtime_cfg = {
        "path": str(runtime_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": build_names_dict(class_names),
    }
    with open(runtime_yaml, "w") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    split_distribution_df = summarize_split_distribution(split_sets, image_class_sets)
    split_distribution_path = runtime_root / "split_distribution.csv"
    split_distribution_df.to_csv(split_distribution_path, index=False)

    meta = {
        "job_name": job_name,
        "label_csv": str(Path(LABEL_CSV).expanduser().resolve()),
        "split_ratios": normalize_split_ratios(CSV_SPLIT_RATIOS),
        "split_seed": SPLIT_SEED,
        "wbf_iou_threshold": WBF_IOU_THRESHOLD,
        "wbf_skip_box_threshold": WBF_SKIP_BOX_THRESHOLD,
        "csv_bbox_format": CSV_BBOX_FORMAT,
        "bbox_format_counts": dict(bbox_format_counts),
        "num_classes": len(class_names),
        "class_names": class_names,
        "csv_samples": len(all_sample_ids),
        "used_samples": len(used_sample_ids),
        "missing_samples": missing,
        "source_size_lookup_count": len(source_size_lookup),
        "split_counts": split_counts,
        "label_counts": label_counts,
    }
    meta_path = runtime_root / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    split_info = {
        "train_pool_images": len(used_sample_ids),
        "train_images": split_counts.get("train", 0),
        "val_images": split_counts.get("val", 0),
        "test_images": split_counts.get("test", 0),
        "train_txt": "",
        "val_txt": "",
        "test_txt": "",
        "split_distribution": str(split_distribution_path),
        "meta": str(meta_path),
    }

    print("Built CSV runtime dataset:", runtime_root)
    print("CSV runtime yaml:", runtime_yaml)
    print("CSV split counts:", split_counts)
    print("CSV label counts:", label_counts)
    print("CSV bbox format counts:", dict(bbox_format_counts))

    return runtime_root, runtime_yaml, runtime_cfg, split_info


def maybe_val(model, yaml_path: Path, split_name: str):
    runtime_params = get_runtime_train_params()
    try:
        metrics = model.val(
            data=str(yaml_path),
            split=split_name,
            imgsz=runtime_params["imgsz"],
            batch=runtime_params["batch"],
            device=DEVICE,
            workers=WORKERS,
            verbose=False,
        )
        return {
            f"{split_name}_precision": float(metrics.box.mp),
            f"{split_name}_recall": float(metrics.box.mr),
            f"{split_name}_map50": float(metrics.box.map50),
            f"{split_name}_map50_95": float(metrics.box.map),
        }
    except Exception as e:
        print(f"Skip {split_name} eval:", e)
        return {
            f"{split_name}_precision": None,
            f"{split_name}_recall": None,
            f"{split_name}_map50": None,
            f"{split_name}_map50_95": None,
        }


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        return str(dst)
    return ""


def save_job_artifacts(job_name: str, run_name: str, best_weight: Path, row: dict, runtime_yaml: Path):
    job_result_dir = RESULTS_ROOT / job_name
    job_result_dir.mkdir(parents=True, exist_ok=True)

    run_dir = RUNS_ROOT / run_name
    last_weight = run_dir / "weights" / "last.pt"
    best_copy = copy_if_exists(best_weight, job_result_dir / "best.pt")
    last_copy = copy_if_exists(last_weight, job_result_dir / "last.pt")
    yaml_copy = copy_if_exists(runtime_yaml, job_result_dir / "data.runtime.yaml")
    train_csv_copy = copy_if_exists(run_dir / "results.csv", job_result_dir / "train_results.csv")
    args_yaml_copy = copy_if_exists(run_dir / "args.yaml", job_result_dir / "train_args.yaml")

    metrics_json_path = job_result_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    pd.DataFrame([row]).to_csv(job_result_dir / "metrics.csv", index=False)

    artifact_info = {
        "job_result_dir": str(job_result_dir),
        "copied_best_weight": best_copy,
        "copied_last_weight": last_copy,
        "copied_runtime_yaml": yaml_copy,
        "copied_train_results_csv": train_csv_copy,
        "copied_train_args_yaml": args_yaml_copy,
        "run_dir": str(run_dir),
        "metrics_json": str(metrics_json_path),
    }
    return artifact_info


def cleanup_after_job():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_stage_and_memory():
    if not KEEP_STAGE:
        clear_local_stage()
    cleanup_after_job()


# =========================
# MAIN LOOP
# =========================
def main():
    if not csv_mode_enabled():
        validate_class_mode()
    ensure_command_exists("rclone")
    runtime_params = get_runtime_train_params()
    jobs = PREPROCESS_JOBS[:QUICK_CHECK_JOB_LIMIT] if QUICK_CHECK_MODE else PREPROCESS_JOBS

    print("Runtime train params:", runtime_params)
    print("Jobs to run:", [job["name"] for job in jobs])

    all_rows = []

    for job in jobs:
        name = job["name"]
        remote_path = resolve_remote_path(job)

        print("\n" + "=" * 80)
        print("START JOB:", name)
        print("REMOTE:", remote_path)

        cleanup_after_job()

        local_download_dir = DOWNLOAD_ROOT / name
        local_extract_dir = EXTRACT_ROOT / name

        source_dataset_root = ensure_local_job_stage(
            remote_path=remote_path,
            local_download_dir=local_download_dir,
            local_extract_dir=local_extract_dir,
        )

        # 3. tìm dataset root + rewrite yaml local
        if str(source_dataset_root).startswith(str(local_extract_dir)):
            repair_extracted_sidecars(source_dataset_root, local_download_dir)

        if csv_mode_enabled():
            dataset_root = source_dataset_root
            base_data_yaml = None
            runtime_dataset_root, runtime_yaml, data_cfg, split_info = prepare_csv_runtime_dataset(
                dataset_root=dataset_root,
                job_name=name,
            )
        else:
            dataset_root, base_data_yaml = ensure_yolo_dataset(source_dataset_root)
            runtime_dataset_root = dataset_root
            runtime_yaml, data_cfg, split_info = prepare_runtime_yaml(dataset_root, original_yaml=base_data_yaml)

        print("SOURCE_DATASET_ROOT:", source_dataset_root)
        print("DATASET_ROOT:", dataset_root)
        print("RUNTIME_DATASET_ROOT:", runtime_dataset_root)
        print("BASE_DATA_YAML:", base_data_yaml)
        print("RUNTIME_YAML:", runtime_yaml)
        if split_info.get("train_txt"):
            print("Train TXT:", split_info["train_txt"])
        if split_info.get("val_txt"):
            print("Val TXT:", split_info["val_txt"])
        if split_info.get("test_txt"):
            print("Test TXT:", split_info["test_txt"])

        train_count = split_info["train_images"]
        val_count = split_info["val_images"]
        test_count = split_info["test_images"]

        print("Train pool images:", split_info["train_pool_images"])
        print("Train images:", train_count)
        print("Val images:", val_count)
        print("Test images:", test_count)

        # 4. train
        run_name = f"yolov8_{name}"
        if csv_mode_enabled():
            run_name = f"{run_name}_csv"
        elif should_use_14class_mode():
            run_name = f"{run_name}_14class"
        if QUICK_CHECK_MODE:
            run_name = f"{run_name}_quickcheck"
        model = YOLO(MODEL_NAME)

        model.train(
            data=str(runtime_yaml),
            epochs=runtime_params["epochs"],
            imgsz=runtime_params["imgsz"],
            batch=runtime_params["batch"],
            workers=WORKERS,
            device=DEVICE,
            project=str(RUNS_ROOT),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        best_weight = RUNS_ROOT / run_name / "weights" / "best.pt"
        print("BEST_WEIGHT:", best_weight)

        # 5. evaluate
        best_model = YOLO(str(best_weight))

        val_metrics = maybe_val(best_model, runtime_yaml, "val")
        test_metrics = {}
        if "test" in data_cfg:
            test_metrics = maybe_val(best_model, runtime_yaml, "test")

        row = {
            "preprocess_name": name,
            "remote_path": remote_path,
            "label_source": "csv" if csv_mode_enabled() else "dataset",
            "label_csv": LABEL_CSV if csv_mode_enabled() else "",
            "class_mode": "csv" if csv_mode_enabled() else CLASS_MODE,
            "source_dataset_root": str(source_dataset_root),
            "dataset_root": str(dataset_root),
            "runtime_dataset_root": str(runtime_dataset_root),
            "base_data_yaml": str(base_data_yaml) if base_data_yaml else "",
            "runtime_yaml": str(runtime_yaml),
            "run_name": run_name,
            "run_dir": str(RUNS_ROOT / run_name),
            "quick_check_mode": QUICK_CHECK_MODE,
            "train_pool_images": split_info["train_pool_images"],
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "split_distribution": split_info.get("split_distribution", ""),
            "runtime_meta": split_info.get("meta", ""),
            "train_epochs": runtime_params["epochs"],
            "train_imgsz": runtime_params["imgsz"],
            "train_batch": runtime_params["batch"],
            "best_weight": str(best_weight),
        }
        row.update(val_metrics)
        row.update(test_metrics)

        artifact_info = save_job_artifacts(
            job_name=name,
            run_name=run_name,
            best_weight=best_weight,
            row=row,
            runtime_yaml=runtime_yaml,
        )
        row.update(artifact_info)
        all_rows.append(row)

        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False)

        print("\nCURRENT SUMMARY")
        print(summary_df)

        # 6. xóa local dataset để nhường disk cho preprocessing kế tiếp
        cleanup_stage_and_memory()

    print("\nDONE")
    print("Saved summary:", SUMMARY_CSV)
    print(pd.read_csv(SUMMARY_CSV))


if __name__ == "__main__":
    main()
