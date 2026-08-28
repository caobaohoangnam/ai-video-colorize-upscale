# AI Video Restoration Suite

Phục hồi, tô màu và nâng cấp độ phân giải video cũ/chất lượng thấp bằng AI — chạy được trên GPU phổ thông (4GB VRAM).

Đóng gói sẵn 3 pipeline chạy nối tiếp nhau trên từng khung hình: **khử nhiễu/khử mờ** (mô hình tự huấn luyện) → **tô màu video đen trắng** (tự động phát hiện) → **nâng cấp độ phân giải** (Real-ESRGAN). Có xử lý theo ngữ cảnh cảnh quay (đêm / cháy nổ / ngoài trời / hoạt hình) và làm mượt màu theo thời gian để chống nhấp nháy.

---

## Chức năng chính

### 1. Sửa video bị khiếm khuyết (mờ, nhèo, nhiễu)
- Mạng nơ-ron dạng U-Net + residual block, tự huấn luyện (`models/network.py`), phân tích **5 khung hình liên tiếp cùng lúc** (temporal window) để khử nhiễu ổn định hơn so với xử lý từng frame riêng lẻ.
- Học kiểu residual: mô hình chỉ học phần "chỉnh sửa" rồi cộng vào khung hình gốc — giữ chi tiết tốt hơn, giảm hiện tượng làm mịn quá đà (over-smoothing).
- Tự động resize khung hình lớn trước khi đưa vào mô hình để chạy an toàn trên GPU 4GB, sau đó khôi phục lại đúng kích thước gốc.
- Đã kiểm định bằng bộ đánh giá riêng (`evaluate.py`): **PSNR ≈ 24.3 dB, SSIM ≈ 0.83** trên tập video test (xem `evaluation_report.csv` và `evaluation_report_charts.png`).

### 2. Chỉnh màu & tô màu video đen trắng
- Tự động phát hiện video đen trắng (đo độ bão hòa trung bình qua nhiều khung mẫu) và tự động tô màu bằng mô hình colorization (Zhang et al., chạy qua OpenCV DNN — không tốn VRAM GPU).
- Tự động bỏ qua bước tô màu với video hoạt hình để giữ đúng phong cách nghệ thuật gốc.
- **Phân tích cảnh theo ngữ cảnh**: nhận diện cảnh đêm, cháy nổ, ngoài trời, hoạt hình và áp bộ điều chỉnh màu tương ứng riêng cho từng loại (vd. cảnh đêm đẩy tông lạnh, cảnh cháy nổ tăng cam/đỏ ở vùng sáng).
- **Làm mượt màu theo thời gian** (EMA trên kênh màu A/B trong không gian LAB) để chống hiện tượng nhấp nháy màu giữa các khung hình liên tiếp — tự reset khi phát hiện chuyển cảnh.
- Chỉnh màu hậu kỳ: cân bằng trắng tự động (Gray World), tăng bão hòa, auto-contrast — có thể bật/tắt qua config.

### 3. Nâng cấp độ phân giải
- Dùng Real-ESRGAN (bản `realesr-general-x4v3`, nhẹ, tối ưu cho GPU VRAM thấp).
- Chọn thẳng độ phân giải đích mong muốn: **720p, 1080p (Full HD), 1440p (2K), 2160p (4K)** — tự tính lại chiều rộng theo đúng tỉ lệ khung hình gốc, nhận input ở bất kỳ độ phân giải nào (144p, 240p, 360p, 480p, 720p...).
- Luôn upscale ở hệ số tối đa rồi resize chính xác về độ phân giải đích, tránh hiện tượng mờ khi nhảy nhiều bậc chất lượng cùng lúc (vd. 144p → 1440p).

### Ngoài ra
- Tự động phát hiện điểm chuyển cảnh (scene cut) để reset bộ nhớ tạm thời, tránh màu/chi tiết của cảnh trước lem sang cảnh sau.
- Tự động ghép lại âm thanh gốc vào video sau xử lý.
- Toàn bộ tham số điều chỉnh qua một file cấu hình duy nhất (`configs/config.yaml`), không cần sửa code.
- Có sẵn script huấn luyện lại mô hình khử nhiễu (`train.py`) trên bộ video sạch của riêng bạn.

---

## Yêu cầu hệ thống

- Python 3.10+
- GPU NVIDIA khuyến nghị (tối thiểu 4GB VRAM); có thể chạy CPU nhưng chậm hơn nhiều
- FFmpeg (dùng để ghép âm thanh)
- Xem chi tiết thư viện trong `requirements.txt` (PyTorch, OpenCV, Real-ESRGAN, BasicSR, PySceneDetect...)

## Cài đặt

```bash
pip install -r requirements.txt
```

Trọng số mô hình **không nằm trong repo Git** (quá lớn so với giới hạn GitHub). Cần chuẩn bị trước khi chạy:
- `weights/video_restoration_net.pth` — mô hình khử nhiễu/khử mờ tự huấn luyện. **[Tải tại đây](#)** (cập nhật link GitHub Release sau khi upload).
- `weights/realesr-general-x4v3.pth` — mô hình nâng cấp độ phân giải Real-ESRGAN. Tải tại [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) (bản `realesr-general-x4v3.pth`).
- `weights/colorization/` — mô hình tô màu đen trắng, **tự động tải về khi chạy lần đầu** (không cần tải thủ công).

## Sử dụng

```bash
python main.py
```

Chương trình sẽ liệt kê video có trong `data/input/`, cho chọn video và độ phân giải đầu ra mong muốn. Kết quả được lưu vào `data/output/`.

Muốn tuỳ biến sâu hơn (tắt tô màu, đổi mức làm mượt màu, bật auto-contrast, đổi kích thước cửa sổ thời gian...), chỉnh trực tiếp trong `configs/config.yaml`.

## Huấn luyện lại mô hình khử nhiễu

Đặt video chất lượng cao vào `data/train/clean/` rồi chạy:

```bash
python train.py
```

Các tham số huấn luyện (batch size, learning rate, số epoch, trọng số các hàm loss...) đều nằm trong `configs/config.yaml`.

## Đánh giá chất lượng

```bash
python evaluate.py
```

Xuất báo cáo PSNR / SSIM / Delta-E theo từng khung hình ra `evaluation_report.csv` và biểu đồ tổng hợp `evaluation_report_charts.png`.

---

## Cấu trúc dự án

```
main.py                  Điểm vào chính — chạy toàn bộ pipeline phục hồi video
train.py                 Huấn luyện lại mô hình khử nhiễu
evaluate.py              Đánh giá chất lượng đầu ra (PSNR/SSIM/Delta-E)
configs/config.yaml      Toàn bộ tham số cấu hình
models/network.py        Kiến trúc mô hình khử nhiễu (VideoRestorationNet)
utils/
  video_utils.py         Đọc video, phát hiện chuyển cảnh, ghép âm thanh
  upscale_utils.py        Tích hợp Real-ESRGAN
  colorize_utils.py        Tô màu video đen trắng
  color_utils.py           Chỉnh màu hậu kỳ + làm mượt màu theo thời gian
  scene_utils.py           Phân tích loại cảnh và áp color hints
  dataset.py / augment.py  Chuẩn bị dữ liệu huấn luyện
weights/                 Trọng số các mô hình
data/                    Video đầu vào/đầu ra, dữ liệu huấn luyện
```

## Giấy phép & bên thứ ba

Dự án sử dụng 3 thành phần pretrained mã nguồn mở: Real-ESRGAN (BSD-3-Clause), BasicSR (Apache 2.0) và mô hình colorization của Zhang et al. (BSD-2-Clause). Cả ba đều cho phép sử dụng thương mại, với điều kiện giữ nguyên thông báo bản quyền gốc khi phân phối lại. Toàn văn các giấy phép này được liệt kê đầy đủ trong [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — cần đính kèm file này khi bán/phân phối sản phẩm.
