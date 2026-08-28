"""
Colorization module — chuyển đổi video đen trắng sang màu.

Sử dụng mô hình Zhang et al. (2016) "Colorful Image Colorization"
chạy qua OpenCV DNN (Caffe backend). Chạy trên CPU, không tốn VRAM.

Pipeline:
  1. Phát hiện tự động video có phải đen trắng không
  2. Nếu có, tải trọng số pretrained (tự động download lần đầu)
  3. Mỗi frame: BGR → LAB → đưa kênh L vào mạng → nhận AB → ghép lại → BGR
"""

import cv2
import numpy as np
import os
import urllib.request

MODEL_DIR = "weights/colorization"

# Multiple fallback URLs for each file
_FILES = {
    "colorization_deploy_v2.prototxt": [
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/colorization_deploy_v2.prototxt",
        "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt",
    ],
    "colorization_release_v2.caffemodel": [
        "https://github.com/richzhang/colorization/releases/download/colorization_weights_v2/colorization_release_v2.caffemodel",
    ],
    "pts_in_hull.npy": [
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/pts_in_hull.npy",
        "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy",
    ],
}

_MANUAL_URLS = {
    "colorization_deploy_v2.prototxt": "https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/colorization_deploy_v2.prototxt",
    "colorization_release_v2.caffemodel": "https://github.com/richzhang/colorization/releases/tag/colorization_weights_v2",
    "pts_in_hull.npy": "https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/pts_in_hull.npy",
}


def _download_weights():
    os.makedirs(MODEL_DIR, exist_ok=True)
    missing = []

    for name, urls in _FILES.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            continue

        downloaded = False
        for url in urls:
            try:
                print(f"Đang tải {name}...")
                urllib.request.urlretrieve(url, path)
                print(f"  Đã tải: {path}")
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            missing.append(name)

    if missing:
        print("\nKhông thể tự động tải các file sau. Vui lòng tải thủ công:")
        for name in missing:
            print(f"  {name}")
            print(f"    → {_MANUAL_URLS[name]}")
        print(f"\nSau khi tải, đặt file vào thư mục: {os.path.abspath(MODEL_DIR)}")
        raise RuntimeError(f"Thiếu file colorization: {missing}")


def load_colorizer():
    """
    Tải và trả về mô hình colorization (OpenCV DNN).
    Tự động download trọng số nếu chưa có.
    """
    _download_weights()

    prototxt   = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
    caffemodel = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    pts_path   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    pts = np.load(pts_path).transpose().reshape(2, 313, 1, 1).astype(np.float32)

    # Gắn cluster centers vào layer cuối của mạng
    class8 = net.getLayerId("class8_ab")
    conv8  = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts]
    net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype=np.float32)]

    print("Đã tải mô hình colorization thành công.")
    return net


def is_grayscale(frame, threshold=5.0):
    """
    Kiểm tra frame có phải đen trắng không bằng cách đo độ bão hòa trung bình.
    Ngưỡng 5.0 là an toàn — video màu thường > 20.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < threshold


def detect_bw_video(cap, sample_count=10):
    """
    Lấy mẫu nhiều frame để xác định video có phải đen trắng không.
    Trả về True nếu phần lớn frame là grayscale.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_count)
    bw_count = 0

    for i in range(sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret and is_grayscale(frame):
            bw_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return bw_count >= (sample_count // 2)


def colorize_frame(net, frame):
    """
    Tô màu 1 frame đen trắng.

    Args:
        net: mô hình OpenCV DNN đã load
        frame: BGR uint8 frame (có thể là grayscale hoặc BGR giả)
    Returns:
        frame BGR uint8 đã được tô màu
    """
    h, w = frame.shape[:2]

    img = frame.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Resize về 224x224 (kích thước đầu vào của mạng), mean-center
    l_input = cv2.resize(l_channel, (224, 224)) - 50.0
    blob = cv2.dnn.blobFromImage(l_input)

    net.setInput(blob)
    ab_dec = net.forward()[0, :, :, :].transpose(1, 2, 0)  # (56, 56, 2)

    # Phóng to AB về kích thước gốc
    ab_resized = cv2.resize(ab_dec, (w, h))

    # Ghép L gốc + AB dự đoán → chuyển về BGR
    colorized_lab = np.concatenate([l_channel[:, :, np.newaxis], ab_resized], axis=2)
    colorized = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    return np.clip(colorized * 255, 0, 255).astype(np.uint8)
