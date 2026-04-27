# Lung YOLOv8 Training Runner

Repo này dùng `train.py` để train YOLOv8 trên nhiều bộ ảnh preprocessing được lưu trên remote `rclone`.

Flow hiện tại là **CSV label mode**:

- ảnh vẫn được kéo từ rclone như trước
- label đọc từ `annotations_all_merged_other_1000nf.csv`
- `No finding` được giữ làm ảnh background với label rỗng
- bbox abnormal được WBF theo từng `(image_id, class_name)`
- split lại `train/val/test = 7/1/2`
- train từng preprocessing job và lưu metrics/weights

## 1. Setup

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Cần có `rclone` và remote đã config:

```bash
rclone listremotes
```

Remote base mặc định:

```bash
RCLONE_REMOTE_BASE="rclone:"
```

Nếu dataset nằm trong folder con:

```bash
export RCLONE_REMOTE_BASE='rclone:some_parent/'
```

## 2. File Label

Mặc định `train.py` tự dùng file nằm cạnh nó:

```text
annotations_all_merged_other_1000nf.csv
```

CSV cần có các cột:

```text
image_id,rad_id,class_name,x_min,y_min,x_max,y_max,original_split,image_width,image_height
```

Ý nghĩa chính:

- `image_id`: dùng để match ảnh trong dataset rclone
- `class_name`: tên class
- `No finding`: không train thành class YOLO, chỉ tạo label rỗng
- `x_min,y_min,x_max,y_max`: bbox pixel theo ảnh gốc
- `image_width,image_height`: size ảnh gốc để normalize bbox đúng
- `original_split`: ưu tiên match ảnh trong folder nguồn tương ứng

Nếu muốn dùng CSV khác:

```bash
export YOLO_LABEL_CSV=/path/to/labels.csv
```

## 3. Dataset Jobs

Mặc định script chạy lần lượt:

```text
clahe
histogram_eq
adaptive_preprocessed
percentile
raw_minmax
rescale_minmax
base_dataset
LUT
expert_window
```

Mỗi job sẽ:

1. copy/reuse/unpack dataset từ rclone
2. đọc CSV label
3. WBF bbox
4. split train/val/test `7/1/2`
5. tạo runtime dataset `.csv_runtime`
6. train YOLO
7. eval `val` và `test`
8. lưu weights/metrics
9. dọn local ảnh nếu `YOLO_KEEP_STAGE=0`

## 4. Chạy Bằng tmux

Workflow khuyến nghị: chạy quick check trước, nếu ổn thì chạy full.

### Bước 1: Quick Check

Quick check chạy 1 job, ít ảnh, 1 epoch để bắt lỗi CSV, rclone, split, label, train loop.

```bash
tmux new -d -s yolo_csv_quick 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_QUICK_CHECK=1 python train.py 2>&1 | tee quickcheck_csv.log'
```

Theo dõi:

```bash
tmux attach -t yolo_csv_quick
tail -f /home/Ubuntu/LUNG_YOLO_CODE/quickcheck_csv.log
```

Nếu quick check xong không có traceback/error và có `CURRENT SUMMARY`, chuyển sang full run.

### Bước 2: Full Run

Full run mặc định **có dọn ảnh sau mỗi job** vì không set `YOLO_KEEP_STAGE=1`.

```bash
tmux new -d -s yolo_csv 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_QUICK_CHECK=0 python train.py 2>&1 | tee full_train_csv.log'
```

Theo dõi:

```bash
tmux attach -t yolo_csv
tail -f /home/Ubuntu/LUNG_YOLO_CODE/full_train_csv.log
```

### Các Lệnh Khác

Nếu muốn giữ ảnh local để debug:

```bash
tmux new -d -s yolo_csv_keep 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_KEEP_STAGE=1 python train.py 2>&1 | tee full_train_csv_keep.log'
```

Nếu muốn kéo/unpack sạch lại từ đầu:

```bash
tmux new -d -s yolo_csv_clean 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 python train.py 2>&1 | tee full_train_csv_clean.log'
```

Theo dõi tmux:

```bash
tmux ls
tmux attach -t yolo_csv
```

Detach mà không kill job:

```text
Ctrl+b rồi nhấn d
```

## 5. Theo Dõi Log

```bash
tail -f /home/Ubuntu/LUNG_YOLO_CODE/full_train_csv.log
```

Tìm lỗi nhanh:

```bash
grep -i "traceback\|error\|failed" /home/Ubuntu/LUNG_YOLO_CODE/full_train_csv.log
```

Check process:

```bash
ps -ef | grep 'python train.py' | grep -v grep
```

Dấu hiệu chạy bình thường:

- còn session trong `tmux ls`
- log vẫn ra dòng mới
- có `START JOB: ...`
- có `Built CSV runtime dataset: ...`
- có `BEST_WEIGHT: ...`
- có `CURRENT SUMMARY`

Sau quick check nên kiểm tra nhanh:

```bash
grep -i "traceback\|error\|failed" /home/Ubuntu/LUNG_YOLO_CODE/quickcheck_csv.log
```

## 6. Output

Work root mặc định:

```text
./yolo_preprocess_runner
```

Output chính:

```text
yolo_preprocess_runner/runs/
yolo_preprocess_runner/results/<job_name>/
yolo_preprocess_runner/training_summary.csv
```

Trong `results/<job_name>/`:

```text
best.pt
last.pt
data.runtime.yaml
metrics.json
metrics.csv
train_results.csv
train_args.yaml
```

Runtime dataset tạm của mỗi job:

```text
dataset_root/.csv_runtime/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.runtime.yaml
  meta.json
  split_distribution.csv
```

`images/` là symlink sang ảnh đã kéo từ rclone. `labels/` là label YOLO sinh từ CSV sau WBF.

## 7. Dọn Dẹp Disk

Mặc định:

```bash
YOLO_KEEP_STAGE=0
```

Sau mỗi job script sẽ xóa:

```text
yolo_preprocess_runner/downloads/
yolo_preprocess_runner/extracted/
```

Vẫn giữ:

```text
yolo_preprocess_runner/runs/
yolo_preprocess_runner/results/
yolo_preprocess_runner/training_summary.csv
```

Muốn giữ ảnh local để debug:

```bash
export YOLO_KEEP_STAGE=1
```

## 8. Quick Check Không Dùng tmux

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
source .venv/bin/activate
YOLO_QUICK_CHECK=1 python train.py 2>&1 | tee quickcheck_csv.log
```

Tùy chỉnh quick check:

```bash
export YOLO_QUICK_CHECK=1
export YOLO_QUICK_CHECK_JOB_LIMIT=1
export YOLO_QUICK_CHECK_TRAIN_SAMPLES=128
export YOLO_QUICK_CHECK_VAL_SAMPLES=32
export YOLO_QUICK_CHECK_TEST_SAMPLES=32
export YOLO_QUICK_CHECK_EPOCHS=1
export YOLO_QUICK_CHECK_IMGSZ=640
export YOLO_QUICK_CHECK_BATCH=4
python train.py
```

## 9. Biến Môi Trường Hay Dùng

```text
YOLO_LABEL_CSV=/path/to/labels.csv
YOLO_WORK_ROOT=/path/to/workdir
YOLO_QUICK_CHECK=1
YOLO_QUICK_CHECK_JOB_LIMIT=1
YOLO_FORCE_RECOPY=1
YOLO_FORCE_REUNPACK=1
YOLO_KEEP_STAGE=1
RCLONE_REMOTE_BASE=rclone:
YOLO_WBF_IOU_THRESHOLD=0.5
```

## 10. Legacy Mode

Flow cũ dùng label có sẵn trong dataset và `YOLO_CLASS_MODE=22/14`. Chỉ dùng khi muốn bỏ CSV label mode.

Tắt CSV mode:

```bash
export YOLO_LABEL_CSV=
```

Chạy legacy 22 class:

```bash
YOLO_LABEL_CSV= YOLO_CLASS_MODE=22 python train.py
```

Chạy legacy 14 class:

```bash
YOLO_LABEL_CSV= YOLO_CLASS_MODE=14 python train.py
```

Trong CSV label mode hiện tại, `YOLO_CLASS_MODE` không có tác dụng.

## 11. Lỗi Thường Gặp

Thiếu `rclone`:

```text
Thiếu lệnh 'rclone'
```

Kiểm tra/cài lại `rclone`, sau đó chạy lại.

Thiếu ảnh từ remote:

```text
CSV missing images: ...
```

Kiểm tra remote dataset có đủ ảnh chưa. Nếu local stage cũ bị lỗi, chạy sạch:

```bash
YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 python train.py
```

CSV thiếu cột:

```text
CSV thiếu cột bắt buộc
```

Kiểm tra lại header CSV, đặc biệt `image_width,image_height`.

## 12. File Chính

```text
train.py
requirements.txt
annotations_all_merged_other_1000nf.csv
README.md
```
