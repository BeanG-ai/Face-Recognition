# Luồng Hoạt Động Của Hệ Thống Nhận Diện Khuôn Mặt

## 1. Tổng Quan Hệ Thống

Hệ thống nhận diện khuôn mặt trên Jetson Orin Nano X đã được tổng hợp thành một ứng dụng thống nhất với ba chế độ hoạt động:
1. **Chế độ đăng ký**: Chụp khuôn mặt ở nhiều góc độ và tạo embedding
2. **Chế độ nhận diện**: Phát hiện và nhận diện khuôn mặt trong thời gian thực
3. **Chế độ xác thực**: Cung cấp xác thực an toàn với khung hướng dẫn và phản hồi thời gian thực

Cả ba chế độ hiện đã được tích hợp vào một hệ thống thống nhất, có thể truy cập thông qua điểm vào `main.py`, cung cấp trải nghiệm người dùng nhất quán và chia sẻ cùng codebase.

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
python main.py --mode authentication --auth-mode single --threshold 0.65

# Chạy xác thực liên tục trong 60 giây
python main.py --mode authentication --auth-mode continuous --duration 60 --threshold 0.65

# Kích hoạt tính năng bảo mật nâng cao
python main.py --mode authentication --auth-mode single --threshold 0.65 --liveness --matches 5

# Chạy ở chế độ liên tục vô thời hạn (cho đến khi dừng thủ công)
python main.py --mode authentication --auth-mode continuous --duration 0 --threshold 0.7
```

**Tùy chọn cấu hình nâng cao**:
```
--mode        : Chế độ hoạt động (registration, recognition, authentication)
--auth-mode   : Chế độ xác thực (single, continuous)
--duration    : Thời gian cho chế độ liên tục tính bằng giây (0 để vô hạn)
--threshold   : Ngưỡng nhận diện (0.0-1.0)
--liveness    : Kích hoạt phát hiện sự sống để tăng cường bảo mật
--matches     : Số lần khớp liên tiếp cần thiết để xác thực
--timeout     : Thời gian chờ xác thực tính bằng giây
--attempts    : Số lần thử xác thực tối đa
--monitor     : Kích hoạt giám sát hiệu suất hệ thống
--report      : Tạo báo cáo xác thực chi tiết
```

## 5. Cấu Trúc Dự Án và Luồng Thực Thi

```
┌─────────────────┐      ┌───────────────────┐      ┌──────────────────────┐
│ CLI Interface   │      │ main.py           │      │ Core Application     │
│ (command line)  ├─────►│ (điểm vào         ├─────►│ (triển khai theo     │
└─────────────────┘      │  thống nhất)      │      │  từng chế độ)        │
                         └───────────────────┘      └──────────┬───────────┘
                                                               │
                          ┌────────────────────────────────────┼────────────────┐
                          │                                    │                │
                     ┌────▼─────────┐                    ┌─────▼──────────┐     │
                     │ Chế độ       │                    │ Chế độ         │     │
                     │ đăng ký      │                    │ nhận diện      │     │
                     └──────┬───────┘                    └────────┬───────┘     │
                            │                                     │             │
                     ┌──────▼───────┐                    ┌────────▼───────┐     │
                     │ HeadPoseModel│                    │ Detector       │     │    ┌───────────────┐
                     │ (chụp nhiều  │                    │ (phát hiện     │     │    │Chế độ         │
                     │  góc mặt)    │                    │  khuôn mặt)    │     ├───►│xác thực       │
                     └──────┬───────┘                    └────────┬───────┘     │    └───────┬───────┘
                            │                                     │             │            │
                            └─────────────┬─────────────┬─────────┘             │      ┌─────▼─────┐
                                          │             │                       │      │ Hệ thống  │
                                    ┌─────▼─────┐ ┌─────▼──────┐                │      │ xác thực  │
                                    │ face_utils│ │ database   │                │      │           │
                                    │ (xử lý    │ │ (lưu trữ   │◄───────────────┘      └───────────┘
                                    │  ảnh mặt) │ │  dữ liệu)  │
                                    └───────────┘ └────────────┘
```

Kiến trúc hệ thống tổng hợp hiện có các tính năng:

- **Điểm vào thống nhất**: `main.py` đóng vai trò là điểm vào duy nhất cho cả ba chế độ hoạt động
- **Các thành phần dùng chung**: Các tiện ích cốt lõi như phát hiện khuôn mặt, truy cập cơ sở dữ liệu và xử lý hình ảnh được chia sẻ giữa tất cả các chế độ
- **Triển khai theo từng chế độ**: Mỗi chế độ (đăng ký, nhận diện, xác thực) có mã chuyên biệt trong khi vẫn tận dụng các thành phần chung
- **Giao diện dòng lệnh**: `face_recognition_cli.py` cung cấp giao diện đơn giản hóa cho các chức năng chính
```

## 6. Sử Dụng

### Sử dụng Giao Diện Dòng Lệnh:

```bash
# Đăng ký người dùng mới
python face_recognition_cli.py register <username>

# Nhận diện khuôn mặt
python face_recognition_cli.py recognize

# Xác thực người dùng (chế độ đơn lẻ)
python face_recognition_cli.py authenticate

# Xác thực liên tục trong 60 giây
python face_recognition_cli.py authenticate --mode continuous --duration 60
```

### Sử dụng lệnh Python trực tiếp:

```bash
# Đăng ký người dùng mới
python main.py --mode registration --username <username>

# Nhận diện khuôn mặt
python main.py --mode recognition

# Xác thực người dùng (chế độ đơn lẻ)
python main.py --mode authentication --auth-mode single

# Xác thực liên tục trong 60 giây
python main.py --mode authentication --auth-mode continuous --duration 60

# Xác thực với tính năng bảo mật nâng cao
python main.py --mode authentication --liveness --matches 5
```

## 7. Triển Khai Trên Jetson Orin Nano X

1. **Thiết lập môi trường**:
   - Cài đặt thư viện cần thiết: `pip install -r requirements.txt`
   - Tạo cấu trúc thư mục: `python setup_project.py`
   - Cho thiết lập đặc thù Jetson: `./jetson_setup.sh`

2. **Tối ưu hóa model**:
   - Tự động sử dụng tăng tốc CUDA khi có sẵn
   - Chuyển đổi TensorRT để cải thiện hiệu suất suy luận
   - Điều chỉnh tham số ngưỡng nhận diện phù hợp với môi trường triển khai

3. **Giám sát hiệu suất**:
   - Giám sát FPS thời gian thực trong quá trình xác thực
   - Theo dõi tài nguyên hệ thống (CPU, bộ nhớ, nhiệt độ)
   - Báo cáo hiệu suất để tối ưu hóa
   ```bash
   # Giám sát hiệu suất hệ thống trong quá trình xác thực
   python main.py --mode authentication --monitor --report
   ```

4. **Tối ưu hóa phần cứng**:
   - Kích hoạt hiệu suất tối đa trên Jetson:
   ```bash
   # Đặt tốc độ xung nhịp tối đa
   sudo jetson_clocks --fan
   
   # Đặt chế độ hiệu suất
   sudo nvpmodel -m 0
   
   # Giám sát nhiệt độ hệ thống
   sudo jtop
   ```


## 8. Sử Dụng Nâng Cao

### Giám Sát Hiệu Suất Hệ Thống

Hệ thống bao gồm các công cụ giám sát hiệu suất được thiết kế đặc biệt cho Jetson Orin Nano X:

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Khởi tạo        │     │ Thu thập số liệu  │     │ Vẽ biểu đồ dữ liệu   │
│ giám sát        ├────►│ hệ thống          ├────►│ hiệu suất            │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Tính năng**:
- Giám sát thời gian thực CPU, bộ nhớ và nhiệt độ
- Theo dõi hiệu suất trong các phiên xác thực
- Trực quan hóa số liệu hệ thống
- Xuất dữ liệu CSV để phân tích

**Sử dụng**:
```bash
# Giám sát hệ thống độc lập
python system_monitor.py --interval 0.5 --output ./monitoring_data

# Giám sát trong quá trình xác thực
python main.py --mode authentication --monitor --report
```

### Báo Cáo Xác Thực

Chế độ xác thực nâng cao có thể tạo các báo cáo chi tiết về các phiên xác thực:

```bash
# Tạo báo cáo xác thực
python main.py --mode authentication --report --output ./reports
```

Điều này tạo ra một báo cáo JSON chứa:
- Cấu hình xác thực
- Kết quả thành công/thất bại
- Thông tin người dùng
- Thời lượng phiên và dấu thời gian
