# Lung YOLOv8 Training Runner

Repo này dùng [train.py](/home/Ubuntu/LUNG_YOLO_CODE/train.py) để train YOLOv8 lần lượt trên nhiều bộ dữ liệu preprocessing được lưu trên remote `rclone`.

Pipeline hiện tại làm các việc sau:

- copy dataset từ remote về local stage nếu cần
- reuse local stage nếu đã có dữ liệu hợp lệ
- tự unpack `.zip`, `.tar`, `.gz`, `.tgz`
- sync thêm sidecar non-archive như `annotations/`
- tự convert COCO JSON sang YOLO labels nếu dataset chưa ở format YOLO
- hỗ trợ 2 mode label:
  - `22` class gốc
  - `14` class official, trong đó 8 class dư sẽ bị loại khỏi runtime dataset
- tách `val` runtime mới từ `images/train`
- dùng `images/val` cũ làm `test`
- train YOLOv8
- eval trên `val` và `test`
- lưu weights, metrics và bảng tổng hợp

## 1. Yêu Cầu

- Ubuntu/Linux
- Python 3.10+
- GPU CUDA nếu muốn train nhanh
- `rclone` đã config remote

## 2. Cài Đặt

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Nếu máy thiếu `venv`:

```bash
sudo apt install python3.10-venv
```

Cài `rclone`:

```bash
curl https://rclone.org/install.sh | sudo bash
```

Kiểm tra remote:

```bash
rclone listremotes
```

## 3. Các Dataset Đang Chạy

Mặc định [train.py](/home/Ubuntu/LUNG_YOLO_CODE/train.py) loop qua:

- `vinbigdata-CLAHE-png`
- `vinbigdata-HistogramEq-png`
- `vinbigdata-adaptive-preprocessed`
- `vinbigdata-percentile-png`
- `vinbigdata-raw_minmax-png`
- `vinbigdata-rescale_minmax-png`
- `vindr_yolo_dataset`
- `vinbigdata-lut-png`
- `vinbigdata-expert-png`

Tên job local tương ứng:

- `clahe`
- `histogram_eq`
- `adaptive_preprocessed`
- `percentile`
- `raw_minmax`
- `rescale_minmax`
- `base_dataset`
- `LUT`
- `expert_window`

Remote base mặc định:

```bash
RCLONE_REMOTE_BASE="rclone:"
```

Nếu remote thật nằm trong một thư mục cha khác:

```bash
export RCLONE_REMOTE_BASE='rclone:some_parent/'
```

## 4. Format Dataset Được Hỗ Trợ

### YOLO format

```text
dataset_root/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

### COCO/export format

```text
dataset_root/
  images/
    train/
    val/
  annotations/
    instances_train.json
    instances_val.json
```

Với COCO/export format, script sẽ tự tạo:

- `labels/train/*.txt`
- `labels/val/*.txt`
- `data.yaml`

## 5. Quy Ước Split Runtime

Trong flow hiện tại:

- `images/train` gốc được xem là train pool
- `images/val` gốc được xem là test thật
- `val` runtime mới được random tách từ `images/train`

Script sinh ra:

- `train.runtime.txt`
- `val.runtime.txt`
- `test.runtime.txt` nếu có test
- `data.runtime.yaml`

Seed tách split hiện tại là `42`.

## 6. Biến Môi Trường Quan Trọng

### Class mode

- `YOLO_CLASS_MODE=22`
  - train theo full 22 class gốc
- `YOLO_CLASS_MODE=14`
  - train theo 14 class official
  - script dựng runtime dataset riêng dưới `dataset_root/.yolo_runtime_14class/`
  - giữ đúng 14 class sau:
    - `Aortic enlargement`
    - `Atelectasis`
    - `Calcification`
    - `Cardiomegaly`
    - `Consolidation`
    - `ILD`
    - `Infiltration`
    - `Lung Opacity`
    - `Nodule/Mass`
    - `Other lesion`
    - `Pleural effusion`
    - `Pleural thickening`
    - `Pneumothorax`
    - `Pulmonary fibrosis`
  - bỏ 8 class dư:
    - `Clavicle fracture`
    - `Edema`
    - `Emphysema`
    - `Enlarged PA`
    - `Lung cavity`
    - `Lung cyst`
    - `Mediastinal shift`
    - `Rib fracture`

### Quick check

- `YOLO_QUICK_CHECK=1`
- `YOLO_QUICK_CHECK_JOB_LIMIT`
- `YOLO_QUICK_CHECK_TRAIN_SAMPLES`
- `YOLO_QUICK_CHECK_VAL_SAMPLES`
- `YOLO_QUICK_CHECK_TEST_SAMPLES`
- `YOLO_QUICK_CHECK_EPOCHS`
- `YOLO_QUICK_CHECK_IMGSZ`
- `YOLO_QUICK_CHECK_BATCH`

### Recovery và local stage

- `YOLO_FORCE_RECOPY=1`
  - xóa local stage của job rồi copy lại từ remote
- `YOLO_FORCE_REUNPACK=1`
  - ép unpack lại từ đầu
- `YOLO_KEEP_STAGE=1`
  - giữ `downloads/` và `extracted/` sau mỗi job để debug

### Work root

- `YOLO_WORK_ROOT=/path/to/workdir`

Mặc định:

```bash
YOLO_WORK_ROOT=./yolo_preprocess_runner
```

## 7. Tham Số Train Mặc Định

Các tham số đang hard-code trong [train.py](/home/Ubuntu/LUNG_YOLO_CODE/train.py):

- `MODEL_NAME="yolov8n.pt"`
- `EPOCHS=50`
- `IMG_SIZE=1024`
- `BATCH=8`
- `WORKERS=4`
- `VAL_SIZE=3000`

Muốn đổi thì sửa trực tiếp file.

## 8. Chạy Quick Check

Quick check 1 job:

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
source .venv/bin/activate
export YOLO_CLASS_MODE=22
export YOLO_QUICK_CHECK=1
python train.py
```

Quick check mặc định:

- 1 job đầu
- 128 ảnh train
- 32 ảnh val
- 32 ảnh test
- 1 epoch
- `imgsz=640`
- `batch=4`

Lưu ý:

- nếu log in ra `Image sizes 640 train, 640 val` thì đó là do đang chạy quick check
- không phải dataset bị resize permanent
- muốn dùng size full theo config chính thì tắt quick check hoặc set:
  - `YOLO_QUICK_CHECK_IMGSZ=1024`

Quick check nhiều job liên tiếp:

```bash
export YOLO_CLASS_MODE=22
export YOLO_QUICK_CHECK=1
export YOLO_QUICK_CHECK_JOB_LIMIT=3
export YOLO_QUICK_CHECK_TRAIN_SAMPLES=128
export YOLO_QUICK_CHECK_VAL_SAMPLES=32
export YOLO_QUICK_CHECK_TEST_SAMPLES=32
export YOLO_QUICK_CHECK_EPOCHS=1
python train.py 2>&1 | tee multi_quickcheck.log
```

Quick check toàn bộ danh sách nhưng vẫn nhẹ:

```bash
export YOLO_CLASS_MODE=22
export YOLO_QUICK_CHECK=1
export YOLO_QUICK_CHECK_JOB_LIMIT=9
export YOLO_QUICK_CHECK_TRAIN_SAMPLES=64
export YOLO_QUICK_CHECK_VAL_SAMPLES=16
export YOLO_QUICK_CHECK_TEST_SAMPLES=16
export YOLO_QUICK_CHECK_EPOCHS=1
python train.py 2>&1 | tee all_jobs_smoke.log
```

Quick check 14 class:

```bash
export YOLO_CLASS_MODE=14
export YOLO_QUICK_CHECK=1
python train.py 2>&1 | tee quickcheck_14class.log
```

## 9. Chạy Full

Chạy full:

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
source .venv/bin/activate
export YOLO_CLASS_MODE=22
python train.py
```

Chạy full 14 class:

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
source .venv/bin/activate
export YOLO_CLASS_MODE=14
python train.py
```

Lưu ý với mode 14 class:

- script tạo runtime dataset tạm dưới:
  - `dataset_root/.yolo_runtime_14class/`
- ảnh được symlink từ dataset gốc
- label được tạo mới theo đúng 14 class official
- dataset gốc 22-class không bị sửa

Chạy full sạch từ đầu:

```bash
YOLO_CLASS_MODE=22 YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 YOLO_KEEP_STAGE=1 python train.py
```

Chạy full sạch từ đầu cho 14 class:

```bash
YOLO_CLASS_MODE=14 YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 YOLO_KEEP_STAGE=1 python train.py
```

Nếu muốn vừa chạy sạch vừa lưu log:

```bash
cd /home/Ubuntu/LUNG_YOLO_CODE
source .venv/bin/activate
YOLO_CLASS_MODE=14 YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 YOLO_KEEP_STAGE=1 python train.py 2>&1 | tee full_train_14class_clean.log
```

Giải thích:

- `FORCE_RECOPY` để copy lại dữ liệu từ remote
- `FORCE_REUNPACK` để unpack lại local stage
- `KEEP_STAGE` để giữ dữ liệu local nếu cần debug

## 10. Chạy Trong tmux

Cài `tmux` nếu chưa có:

```bash
sudo apt update
sudo apt install -y tmux
```

Chạy full nền và ghi log:

```bash
tmux new -d -s yolo_full 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_CLASS_MODE=22 python train.py 2>&1 | tee full_train.log'
```

Chạy full 14 class trong `tmux`:

```bash
tmux new -d -s yolo_full_14 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_CLASS_MODE=14 python train.py 2>&1 | tee full_train_14class.log'
```

Chạy full nhưng giữ local stage:

```bash
tmux new -d -s yolo_full_keep 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_CLASS_MODE=22 YOLO_KEEP_STAGE=1 python train.py 2>&1 | tee full_train_keep.log'
```

Chạy full sạch từ đầu:

```bash
tmux new -d -s yolo_full_clean 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_CLASS_MODE=22 YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 YOLO_KEEP_STAGE=1 python train.py 2>&1 | tee full_train_clean.log'
```

Chạy full sạch từ đầu cho 14 class trong `tmux`:

```bash
tmux new -d -s yolo_full_14_clean 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_CLASS_MODE=14 YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 YOLO_KEEP_STAGE=1 python train.py 2>&1 | tee full_train_14class_clean.log'
```

Chạy quick check trong `tmux`:

```bash
tmux new -d -s yolo_quickcheck 'cd /home/Ubuntu/LUNG_YOLO_CODE && source .venv/bin/activate && YOLO_CLASS_MODE=22 YOLO_QUICK_CHECK=1 python train.py 2>&1 | tee quickcheck.log'
```

Lệnh hay dùng:

```bash
tmux ls
tmux attach -t yolo_full
```

Detach mà không kill job:

- nhấn `Ctrl+b`
- thả ra
- nhấn `d`

## 11. Check Log Và Process

Nếu chạy với `tee`:

```bash
tail -f /home/Ubuntu/LUNG_YOLO_CODE/full_train.log
tail -f /home/Ubuntu/LUNG_YOLO_CODE/quickcheck.log
tail -f /home/Ubuntu/LUNG_YOLO_CODE/full_train_14class.log
```

Xem 100 dòng cuối:

```bash
tail -n 100 /home/Ubuntu/LUNG_YOLO_CODE/full_train.log
```

Tìm lỗi nhanh:

```bash
grep -i "traceback\\|error\\|failed" /home/Ubuntu/LUNG_YOLO_CODE/full_train.log
```

Check process:

```bash
ps -ef | grep 'python train.py' | grep -v grep
```

Dấu hiệu job đang chạy bình thường:

- còn thấy session trong `tmux ls`
- log vẫn ra dòng mới
- có `START JOB: ...`
- có `BEST_WEIGHT: ...`
- có `CURRENT SUMMARY`

## 12. Output Được Lưu Ở Đâu

Work root mặc định:

```bash
./yolo_preprocess_runner
```

Các output chính:

- `./yolo_preprocess_runner/runs/`
  - output gốc của Ultralytics
- `./yolo_preprocess_runner/results/<job_name>/`
  - `best.pt`
  - `last.pt`
  - `data.runtime.yaml`
  - `metrics.json`
  - `metrics.csv`
  - `train_results.csv`
  - `train_args.yaml`
- `./yolo_preprocess_runner/training_summary.csv`
  - bảng tổng hợp tất cả job đã chạy trong phiên hiện tại
- nếu chạy `YOLO_CLASS_MODE=14`, mỗi dataset sẽ có runtime root riêng:
  - `dataset_root/.yolo_runtime_14class/`
  - đây là dataset tạm đã lọc về đúng 14 class
  - `images/` là symlink sang ảnh gốc
  - `labels/` là label 14-class mới được sinh ra

## 13. Sau Khi Chạy Xong Nó Xóa Gì

Nếu:

```bash
YOLO_KEEP_STAGE=0
```

thì sau mỗi job script sẽ xóa:

- `WORK_ROOT/downloads`
- `WORK_ROOT/extracted`

Nhưng vẫn giữ:

- `WORK_ROOT/runs`
- `WORK_ROOT/results`
- `WORK_ROOT/training_summary.csv`

Ngoài ra script chỉ dọn thêm:

- Python GC qua `gc.collect()`
- CUDA cache qua `torch.cuda.empty_cache()`

Nếu muốn giữ local stage để debug:

```bash
export YOLO_KEEP_STAGE=1
```

## 14. Flow Reuse Và Recovery Local Stage

Trước khi copy lại từ remote, script sẽ thử theo thứ tự:

1. reuse `extracted/` nếu đã có dataset root
2. sync sidecar non-archive từ `downloads/` sang `extracted/`
3. dùng dataset trực tiếp từ `downloads/` nếu đủ cấu trúc
4. unpack archive từ `downloads/` sang `extracted/`
5. nếu local không usable thì recopy từ remote

Code hiện tại đã cover các case phổ biến:

- thiếu `annotations`
- partial unpack
- remote copy xong nhưng local chưa bung archive
- COCO `file_name` dạng `images/train/xxx.png`

## 15. Lỗi Thường Gặp

### Thiếu `rclone`

Script sẽ fail sớm nếu máy chưa có `rclone`.

### Thiếu `instances_train.json` hoặc `instances_val.json`

Dataset COCO/export cần đủ:

```text
annotations/instances_train.json
annotations/instances_val.json
```

### Thiếu ảnh được tham chiếu trong COCO JSON

Nếu lỗi kiểu:

```text
FileNotFoundError: Thiếu ảnh images/train/xxx.png trong ...
```

thì nghĩa là local stage hoặc remote source đang thiếu file ảnh thật. Khi đó nên chạy sạch:

```bash
YOLO_FORCE_RECOPY=1 YOLO_FORCE_REUNPACK=1 YOLO_KEEP_STAGE=1 python train.py
```

Nếu vẫn lỗi cùng một ảnh sau khi chạy sạch, vấn đề nằm ở source data chứ không phải `tmux`.

## 16. File Chính Trong Repo

- [train.py](/home/Ubuntu/LUNG_YOLO_CODE/train.py)
  - pipeline train chính
- [requirements.txt](/home/Ubuntu/LUNG_YOLO_CODE/requirements.txt)
  - dependency runtime
- [README.md](/home/Ubuntu/LUNG_YOLO_CODE/README.md)
  - tài liệu sử dụng
