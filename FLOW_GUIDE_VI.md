# Luồng Hoạt Động Của Hệ Thống Nhận Diện Khuôn Mặt

## 1. Tổng Quan Hệ Thống

Hệ thống nhận diện khuôn mặt trên Jetson Orin Nano X bao gồm ba luồng chính:
1. **Luồng đăng ký người dùng mới**: Chụp khuôn mặt ở nhiều góc độ và tạo embedding
2. **Luồng nhận diện khuôn mặt**: Phát hiện và nhận diện khuôn mặt trong thời gian thực
3. **Luồng xác thực khuôn mặt**: Cung cấp xác thực an toàn với khung hướng dẫn và phản hồi thời gian thực

## 2. Luồng Đăng Ký Người Dùng Mới

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Khởi tạo        │     │ Hiển thị hướng dẫn│     │ Quay mặt người dùng  │
│ camera + UI     ├────►│ xoay mặt (4 góc)  ├────►│ (trái/phải/lên/thẳng)│
└─────────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                             │
┌─────────────────┐     ┌───────────────────┐     ┌──────────▼───────────┐
│ Lưu thông tin   │     │ Trích xuất        │     │ Lưu ảnh mặt đã chụp  │
│ người dùng vào  │◄────┤ embedding từ ảnh  │◄────┤ theo từng góc quay   │
│ cơ sở dữ liệu   │     │ đã chụp           │     │                      │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Chi tiết kỹ thuật**:
1. Sử dụng class `HeadPoseEnrollment` để hướng dẫn người dùng quay mặt theo 4 hướng
2. Dùng `MediaPipe FaceMesh` để ước tính góc quay của đầu (yaw, pitch)
3. Khi đạt đủ góc quay, hệ thống sẽ tự động chụp ảnh khuôn mặt
4. Dùng model `inception_resnet_v1.onnx` để trích xuất embedding từ các ảnh đã chụp
5. Tính trung bình các embedding để tạo ra vector đặc trưng cho người dùng
6. Lưu embedding và thông tin người dùng vào cơ sở dữ liệu

## 3. Luồng Nhận Diện Khuôn Mặt Thời Gian Thực

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Khởi tạo camera │     │ Phát hiện khuôn   │     │ Cắt và xử lý         │
│ + detector      ├────►│ mặt trong frame   ├────►│ ảnh khuôn mặt        │
└─────────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                             │
┌─────────────────┐     ┌───────────────────┐     ┌──────────▼───────────┐
│ Hiển thị kết quả│     │ So sánh với       │     │ Trích xuất embedding │
│ lên màn hình    │◄────┤ database và       │◄────┤ từ khuôn mặt         │
│                 │     │ tính độ tương đồng│     │                      │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Chi tiết kỹ thuật**:
1. Sử dụng `FaceDetector` (MediaPipe) để phát hiện khuôn mặt trong frame video
2. Với mỗi khuôn mặt phát hiện được:
   - Cắt vùng khuôn mặt từ frame
   - Tiền xử lý ảnh (resize, chuẩn hóa)
   - Trích xuất embedding sử dụng model `inception_resnet_v1.onnx`
3. So sánh embedding với database:
   - Tính cosine similarity giữa embedding hiện tại và các embedding đã lưu
   - Nếu độ tương đồng vượt ngưỡng (threshold), xác định người dùng
4. Hiển thị kết quả:
   - Vẽ bounding box quanh khuôn mặt
   - Hiển thị tên người dùng nếu nhận diện thành công
   - Hiển thị "Unknown" nếu không nhận diện được

## 4. Luồng Xác Thực Khuôn Mặt

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Khởi tạo hệ     │     │ Hiển thị khung    │     │ Cung cấp phản hồi    │
│ thống xác thực  ├────►│ hướng dẫn         ├────►│ vị trí thời gian thực│
└─────────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                             │
┌─────────────────┐     ┌───────────────────┐     ┌──────────▼───────────┐
│ Hiển thị kết    │     │ So sánh với cơ sở │     │ Chụp và xử lý khuôn  │
│ quả xác thực    │◄────┤ dữ liệu (kiểm tra │◄────┤ mặt khi đã đúng vị   │
│                 │     │ nhiều lần)        │     │ trí                  │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Chi tiết kỹ thuật**:
1. Sử dụng lớp `FacialAuthenticationSystem` để hướng dẫn người dùng qua quá trình xác thực
2. Cung cấp **hai chế độ**:
   - **Xác thực đơn lẻ**: Xác minh một lần để truy cập an toàn
   - **Xác thực liên tục**: Giám sát thời gian thực để xác minh liên tục
3. Tính năng:
   - **Giao diện khung hướng dẫn**: Hướng dẫn trực quan giúp người dùng định vị khuôn mặt
   - **Phản hồi vị trí thời gian thực**: Hướng dẫn di chuyển (trái/phải/lên/xuống/gần hơn/xa hơn)
   - **Xác thực tiến dần**: Yêu cầu nhiều lần nhận diện thành công liên tiếp để tăng cường bảo mật
   - **Phản hồi trực quan**: Giao diện mã màu hiển thị trạng thái xác thực
   - **Hiệu ứng chào mừng**: Xác nhận trực quan khi xác thực thành công

### Chế độ xác thực đơn lẻ
- Hướng dẫn người dùng định vị khuôn mặt đúng cách
- Yêu cầu nhiều lần khớp thành công liên tiếp để xác thực
- Trả về kết quả xác thực (thành công/thất bại) và thông tin người dùng

### Chế độ xác thực liên tục
- Giám sát khuôn mặt liên tục trong khoảng thời gian xác định
- Có thể nhận diện nhiều người dùng trong một phiên
- Cung cấp phản hồi thời gian thực về trạng thái xác thực
- Trả về danh sách tất cả người dùng đã được xác thực trong phiên

**Ví dụ sử dụng**:
```bash
# Chạy xác thực khuôn mặt (chế độ đơn lẻ)
python main.py --mode authentication 

# Chạy xác thực liên tục trong 60 giây
python main.py --mode authentication --auth-mode continuous --duration 60
```

## 5. Cấu Trúc Dự Án và Luồng Thực Thi

```
┌─────────────────┐      ┌───────────────────┐      ┌──────────────────────┐
│ CLI Interface   │      │ main.py           │      │ Project/run.py       │
│ (command line)  ├─────►│ (entry point)     ├─────►│ (core application)   │
└─────────────────┘      └───────────────────┘      └──────────┬───────────┘
                                                               │
                          ┌────────────────────────────────────┼────────────────┐
                          │                                    │                │
                     ┌────▼─────────┐                    ┌─────▼──────────┐     │
                     │ registration │                    │ recognition    │     │
                     │ (đăng ký)    │                    │ (nhận diện)    │     │
                     └──────┬───────┘                    └────────┬───────┘     │
                            │                                     │             │
                     ┌──────▼───────┐                    ┌────────▼───────┐     │
                     │ HeadPoseModel│                    │ Detector       │     │
                     │ (chụp nhiều  │                    │ (phát hiện     │     │
                     │  góc mặt)    │                    │  khuôn mặt)    │     │
                     └──────┬───────┘                    └────────┬───────┘     │
                            │                                     │             │
                            └─────────────┬─────────────┬─────────┘             │
                                          │             │                       │
                                    ┌─────▼─────┐ ┌─────▼──────┐                │
                                    │ face_utils│ │ database   │                │
                                    │ (xử lý    │ │ (lưu trữ   │◄───────────────┘
                                    │  ảnh mặt) │ │  dữ liệu)  │
                                    └───────────┘ └────────────┘
```

## 6. Triển Khai Trên Jetson Orin Nano X

1. **Thiết lập môi trường**:
   - Cài đặt thư viện cần thiết: `pip install -r requirements.txt`
   - Tạo cấu trúc thư mục: `python setup_project.py`

2. **Tối ưu hóa model** (tuỳ chọn):
   - Chuyển đổi models sang TensorRT để tăng hiệu suất
   - Điều chỉnh tham số ngưỡng nhận diện phù hợp với môi trường thực tế

3. **Chạy ứng dụng**:
   - Sử dụng các lệnh ở mục 5
   - Có thể cài đặt thành dịch vụ tự động khởi động khi boot
