"""
Color correction post-processing cho inference pipeline.

Các hàm này chạy trên CPU (OpenCV/NumPy) nên KHÔNG tốn VRAM.
Được gọi SAU khi model đã khử nhiễu, TRƯỚC khi upscale.
"""

import cv2
import numpy as np


class TemporalColorSmoother:
    """
    Làm mượt màu sắc theo thời gian giữa các frame liên tiếp.

    Hoạt động trong không gian LAB:
      - Kênh L (độ sáng) giữ nguyên — không ảnh hưởng đến độ nét
      - Kênh A, B (màu sắc) được blend với frame trước bằng EMA

    alpha gần 1.0 = ưu tiên frame hiện tại (ít làm mượt)
    alpha gần 0.5 = làm mượt mạnh (màu ổn định hơn, phản ứng chậm hơn)
    """

    def __init__(self, alpha=0.75):
        self.alpha = alpha
        self.prev_ab = None  # AB channels của frame trước

    def smooth(self, frame):
        """
        Args:
            frame: BGR uint8 frame
        Returns:
            frame BGR uint8 với màu đã được làm mượt theo thời gian
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_ch = lab[:, :, 0]
        ab_ch = lab[:, :, 1:]

        if self.prev_ab is None or self.prev_ab.shape != ab_ch.shape:
            # Frame đầu tiên hoặc sau scene cut — không blend
            self.prev_ab = ab_ch.copy()
            return frame

        # Exponential Moving Average trên kênh màu A, B
        smoothed_ab = self.alpha * ab_ch + (1.0 - self.alpha) * self.prev_ab
        self.prev_ab = smoothed_ab.copy()

        # Ghép lại L + smoothed AB
        lab[:, :, 1:] = smoothed_ab
        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def reset(self):
        """Gọi khi phát hiện chuyển cảnh để tránh blend màu cảnh cũ sang cảnh mới."""
        self.prev_ab = None


def auto_white_balance(frame):
    """
    Cân bằng trắng tự động bằng thuật toán Gray World.
    Giả định: trung bình màu của toàn bộ ảnh nên gần với xám trung tính.
    Nhẹ, nhanh, phù hợp chạy mỗi frame.
    """
    img = frame.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img, axis=(0, 1))
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Scale mỗi kênh để trung bình bằng nhau
    if avg_b > 0:
        img[:, :, 0] *= avg_all / avg_b
    if avg_g > 0:
        img[:, :, 1] *= avg_all / avg_g
    if avg_r > 0:
        img[:, :, 2] *= avg_all / avg_r

    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_saturation(frame, factor=1.15):
    """
    Tăng/giảm độ bão hòa.
    factor > 1.0 = màu rực hơn, < 1.0 = nhạt hơn.
    Khử nhiễu thường làm mất bão hòa nhẹ, nên tăng 10-20% là hợp lý.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def histogram_match_lab(source, reference):
    """
    Khớp histogram trong không gian LAB.
    Dùng khi muốn frame đầu ra có tone màu giống frame tham chiếu.
    Chỉ khớp kênh A và B (màu sắc), giữ nguyên kênh L (độ sáng).
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Khớp mean + std cho kênh A và B
    for ch in [1, 2]:
        src_mean, src_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std()
        ref_mean, ref_std = ref_lab[:, :, ch].mean(), ref_lab[:, :, ch].std()

        if src_std > 0:
            src_lab[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean

    src_lab = np.clip(src_lab, 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def auto_contrast(frame, clip_percent=1.0):
    """
    Tự động kéo giãn tương phản (histogram stretching).
    clip_percent: phần trăm pixel bị cắt ở hai đầu histogram.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    # Tính ngưỡng cắt
    total_pixels = l_channel.size
    low_count = int(total_pixels * clip_percent / 100.0)
    high_count = int(total_pixels * (1.0 - clip_percent / 100.0))

    sorted_vals = np.sort(l_channel.flatten())
    low_val = sorted_vals[low_count]
    high_val = sorted_vals[high_count]

    if high_val > low_val:
        l_channel = (l_channel - low_val) / (high_val - low_val) * 255.0
        l_channel = np.clip(l_channel, 0, 255)

    lab[:, :, 0] = l_channel.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_color_correction(frame, config):
    """
    Pipeline chỉnh màu tổng hợp — gọi từ main.py.

    config: dict chứa các toggle, ví dụ:
        {
            'white_balance': True,
            'saturation_factor': 1.15,
            'auto_contrast': True,
            'contrast_clip': 1.0
        }
    """
    result = frame

    if config.get('white_balance', False):
        result = auto_white_balance(result)

    sat_factor = config.get('saturation_factor', 0)
    if sat_factor > 0 and sat_factor != 1.0:
        result = adjust_saturation(result, factor=sat_factor)

    if config.get('auto_contrast', False):
        clip = config.get('contrast_clip', 1.0)
        result = auto_contrast(result, clip_percent=clip)

    return result
