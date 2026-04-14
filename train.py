import os
from pathlib import Path
import gc
import json
import shutil as shutil_lib
import shutil
import subprocess
import zipfile

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
IMG_SIZE = 1024
BATCH = 8
WORKERS = 4

SPLIT_SEED = 42
VAL_SIZE = 3000

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


def sanitize_class_name(name: str):
    return str(name).strip()


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

    if data_yaml.exists() and labels_dir.exists():
        return

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


def prepare_runtime_yaml(dataset_root: Path):
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

    runtime_train_files = [
        str(p.resolve()) for i, p in enumerate(train_pool_files) if i not in val_indices
    ]
    runtime_val_files = [
        str(p.resolve()) for i, p in enumerate(train_pool_files) if i in val_indices
    ]
    runtime_test_files = [
        str(p.resolve()) for p in list_image_files(test_dir)
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

        dataset_root = ensure_local_job_stage(
            remote_path=remote_path,
            local_download_dir=local_download_dir,
            local_extract_dir=local_extract_dir,
        )

        # 3. tìm dataset root + rewrite yaml local
        if str(dataset_root).startswith(str(local_extract_dir)):
            repair_extracted_sidecars(dataset_root, local_download_dir)
        ensure_yolo_dataset(dataset_root)
        runtime_yaml, data_cfg, split_info = prepare_runtime_yaml(dataset_root)

        print("DATASET_ROOT:", dataset_root)
        print("RUNTIME_YAML:", runtime_yaml)
        print("Train TXT:", split_info["train_txt"])
        print("Val TXT:", split_info["val_txt"])
        if split_info["test_txt"]:
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
            "dataset_root": str(dataset_root),
            "runtime_yaml": str(runtime_yaml),
            "run_name": run_name,
            "run_dir": str(RUNS_ROOT / run_name),
            "quick_check_mode": QUICK_CHECK_MODE,
            "train_pool_images": split_info["train_pool_images"],
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
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
