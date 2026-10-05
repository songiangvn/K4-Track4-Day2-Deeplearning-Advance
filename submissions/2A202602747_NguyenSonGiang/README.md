# Lab Day 2: DeepWeeds, bài nộp của Nguyễn Sơn Giang (2A202602747)

Kết quả chính (test fold 0, 3 seed, tính bằng `eval.py` của repo gốc):

| Cấu hình | macro-F1 test | top-1 test | Ghi chú |
|---|---|---|---|
| **F01** Swin-T (res 288, 20 epoch) + 10 crop + temperature scaling | **0,9829 ± 0,0007** | **98,64%** | tốt nhất, chạy ngoại tuyến; tự chấm phần I: 19/20 |
| F03 MobileNetV3-L + KD + 1 view 352 + TS | 0,9693 ± 0,0010 | 97,60% | thời gian thực: p95 4,35 ms (batch 1, RTX 3090) |
| T00 mốc Swin-T (công thức nền + 1 view) | 0,9701 ± 0,0008 | 97,76% | mốc để so mức cải thiện |

Phân tích đầy đủ trong [`report.md`](report.md). Mọi bảng số nằm trong [`results.xlsx`](results.xlsx).

## Nội dung thư mục

| Đường dẫn | Nội dung |
|---|---|
| `report.md` | Báo cáo (dàn ý GUIDE mục 6.3) |
| `results.xlsx` | Sheet `Summary`, `Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, và thêm `MethodSelect`, `ONNX`, `Robustness` |
| `curves/` | Đường cong training của **mọi** thí nghiệm (86 ảnh, `<exp_id>_<mô tả>.png`; ảnh vẽ từ seed chạy sau cùng) |
| `predictions/` | Dự đoán test và val của chung kết F01/F02/F03 và mốc T00/T00m, mỗi seed một file; bản chưa TS là `*uncal*` |
| `eval/` | Đầu ra của `eval.py score` và `eval.py grade` |
| `figures/` | EDA, kiểm tra pipeline, biểu đồ tổng hợp, Grad-CAM, đánh đổi độ chính xác và độ trễ, lệch phân phối |
| `eda_split_check.json`, `sanity.json` | Số liệu kiểm tra chia dữ liệu và kiểm tra pipeline |
| `requirements.txt` | Phiên bản thư viện, ghim chính xác |
| `code/` | Toàn bộ code (bên dưới) |

### Code

| File | Vai trò |
|---|---|
| `dataset.py`, `model.py`, `losses.py`, `train.py`, `inference.py`, `benchmark.py` | Bộ khung `starter/` đã hoàn thiện, giữ nguyên tên hàm và giao diện vào/ra |
| `run_exps.py` | Danh sách **mọi** thí nghiệm (exp_id → `Config`) và quyết định chọn trên val (ghi chú ngay trong code). Mỗi thí nghiệm chạy trong một process con riêng |
| `eda.py`, `sanity.py` | Bước 0: kiểm tra chia dữ liệu, EDA, kiểm tra pipeline |
| `run_inference.py`, `method_select.py` | Bước 3: so sánh phương pháp suy luận trên val (1 model, có đo độ trễ / 3 seed) |
| `final_predict.py` | Bước 4: **nơi duy nhất chạm vào test**, chạy đúng một lần cho mỗi seed (có file khoá) |
| `make_results.py`, `make_figures.py` | Tạo `results.xlsx` và các biểu đồ tổng hợp |
| `res_sweep.py`, `robustness.py`, `gradcam.py`, `onnx_bench.py` | Chẩn đoán và điểm thưởng |
| `run_seq.py` | Chạy nhiều script lần lượt trong một job SLURM |
| `test_code.py` | 22 unit test (`python -m unittest test_code`) |
| `lab_day2.ipynb` | Notebook tổng hợp: tải dữ liệu, kiểm tra, đọc kết quả, vẽ, chạy `eval.py` |

## Môi trường

- Python 3.10.19, torch 2.6.0+cu124, torchvision 0.21.0, timm 1.0.15, numpy 1.26.4, scikit-learn 1.7.2. Đầy đủ trong `requirements.txt`.
- GPU: cụm SLURM với RTX 3090 / RTX 4090 (24 GB). Mỗi số đo độ trễ đều ghi kèm tên GPU.
- **Notebook:** `code/lab_day2.ipynb`. Bài được chạy trên cụm GPU SLURM của trường, **không chạy trên Colab/Kaggle**, nên không có link Colab. Notebook vẫn chạy được trên Colab/Kaggle: ô đầu tải dữ liệu và cài thư viện; đặt `RUN_TRAINING = True` để train lại một thí nghiệm mẫu.

```bash
conda create -p <env> python=3.10.19 -y && conda activate <env>
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Dữ liệu

Đặt dữ liệu ở `<repo>/data/`, **không commit**:
- `data/images/*.jpg`: ảnh lấy từ `images.zip` trên Zenodo (MD5 `b7b30f96d466fba86016aa5a26606e0f`).
- `data/labels/`: các file `labels.csv`, `{train,val,test}_subset0.csv` lấy từ GitHub của tác giả.

Lần chạy đầu sẽ tự tạo `data/images_bytes.pkl`, chứa byte JPEG gốc gom vào một file. File này giúp đọc dữ liệu nhanh hơn qua ổ mạng và không làm thay đổi ảnh.

## Chạy lại, theo thứ tự

Chạy mọi lệnh từ `code/`. Trên SLURM, các script tương ứng nằm trong `<repo>/slurm/`, ví dụ `sbatch run_exps.sh B2 --seeds 0 1 2`.

```bash
python -m unittest test_code                      # 22 test, chạy được không cần GPU
python eda.py && python sanity.py                 # Bước 0
python run_exps.py B  --seeds 0                   # Bước 1, công thức nền gốc (B01-B07)
python run_exps.py S  --seeds 0 1 2               # ablation scale của crop (chẩn đoán)
python run_exps.py B2 X --seeds 0 1 2             # Bước 1, công thức nền đã sửa (B11-B17) + DINOv2 (X01-X03)
python run_exps.py Ts Tm --seeds 0 1 2            # Bước 2: Swin-T và MobileNetV3-L, 26 thí nghiệm mỗi backbone
python run_exps.py Cs Cm KD --seeds 0 1 2         # kết hợp + chưng cất tri thức
python run_exps.py F --seeds 3 4 5                # chung kết F01, F02 (seed mới)
python run_inference.py --run C01s/seed0 --ensemble C01s/seed1 C01s/seed2 --tag I_C01s    # Bước 3 (val)
python method_select.py --runs C01s/seed0 C01s/seed1 C01s/seed2 --tag MS_C01s            # (tương tự cho C03m, X11)
# Bước 4: test, đúng một lần cho mỗi seed (xem slurm/submit_final.sh để có đủ danh sách)
python final_predict.py --run F01/seed3 --method I02bL --ts --exp-id F01
python final_predict.py --run X11/seed0 --method I04_f352 --ts --exp-id F03
python final_predict.py --run T00s/seed0 --method I00 --exp-id T00
python robustness.py --run F02/seed3 && python gradcam.py --run F01/seed3 --split test
python onnx_bench.py --run F02/seed3 --fuse-bn
python make_results.py --final F01 F02 F03 --baseline T00 T00m && python make_figures.py
cd <repo> && python eval.py score --pred "submissions/2A202602747_NguyenSonGiang/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01
```

**Seed:**
- Bước 1–2 và các tổ hợp kết hợp dùng seed 0, 1, 2.
- Chung kết F01, F02 dùng seed **3, 4, 5**, mới hoàn toàn so với lúc chọn cấu hình.
- F03 dùng 3 seed của X11 (0, 1, 2), như đã ghi trong báo cáo.
- `cudnn.benchmark = True`, nên chạy lại cho kết quả cùng mức nhưng không giống từng bit.

**Không commit:** dataset, `data/images_bytes.pkl`, checkpoint (`runs/**/best.pt`), file `.onnx`. Checkpoint có thể chia sẻ khi được yêu cầu.
