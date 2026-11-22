# Báo cáo rà soát cấu trúc repo

Ngày: 2026-11-22 · Phạm vi: chỉ rà soát và dọn file, **không sửa code**.

## 1. Đã thực hiện
- `main` được đặt bằng nội dung của `feature/merged-ui-vu`; xóa 8 nhánh còn lại. Bản `main` cũ được giữ ở tag `backup/old-main`.
- Xóa các file YOLO không còn dùng:
  - `Project/models/yolov10s-face.onnx` (28 MB)
  - `Project/models/yolov11m-face(int8).engine` (25 MB)
  - `Project/models/yolov11m-face.pt` (39 MB)
  - `Project/utils/Detector(YOLO).py`
- Kiểm tra trước khi xóa: không file Python nào import `Detector(YOLO)` (tên có dấu ngoặc nên không import được); `recognition.py` và `registration.py` dùng `Project.utils.Detector` (MediaPipe). Không có `ultralytics` trong `requirements.txt`.

## 2. Việc còn tồn đọng (chưa sửa)
| Mục | Chi tiết |
|---|---|
| Tham chiếu YOLO trong script | `setup.sh` (dòng ~226-235, 429-430) và `jetson_setup.sh` (~457-467, 632-633) vẫn kiểm tra file YOLO. Chỉ là chẩn đoán, sẽ in "Missing" / "not found (optional)", không làm lỗi. |
| Bug có sẵn | `main_menu.py:225` gọi `optimize_models.py` nhưng file này không tồn tại (chỉ có `optimize_models.sh` và `.ps1`). |
| `authentication.py` | 1.536 dòng, nên tách module. |
| Model trong git | `inception_resnet_v1.onnx` (90 MB) và các file `.pth` vẫn được commit; nên dùng Git LFS hoặc script tải về. |
| Thư mục gốc | 15 file lẫn lộn (entry point, setup, optimize, test, tài liệu). |

## 3. Đề xuất sắp xếp (cần sửa đường dẫn/import nên làm riêng, rồi chạy thử)
- `docs/`: `CAMERA_CONFIG.md`, `FLOW_GUIDE_EN.md`, `FLOW_GUIDE_VI.md` (README.md đang link tới chúng, cần cập nhật link).
- `scripts/`: `setup.sh`, `setup.ps1`, `jetson_setup.sh`, `optimize_models.*`, `optimize_*.py` (README và các script gọi nhau bằng đường dẫn gốc, cần sửa).
- `tests/`: `test_camera_config.py`.
- `tools/`: `system_monitor.py` (`main.py` và `main_menu.py` đang tham chiếu).
