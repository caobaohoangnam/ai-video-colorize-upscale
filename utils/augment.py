"""
Synthetic degradation pipeline cho training.
Tạo dữ liệu đầu vào "hỏng" (nhiễu + sai màu) từ frame sạch,
để mô hình học cách khử nhiễu VÀ chỉnh màu cùng lúc.
"""

import cv2
import numpy as np


def add_gaussian_noise(img, sigma_range=(5, 50)):
    """Thêm nhiễu Gaussian (giả lập nhiễu camera ISO cao)."""
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_poisson_noise(img, scale_range=(1.0, 5.0)):
    """Thêm nhiễu Poisson (giả lập nhiễu photon trong điều kiện thiếu sáng)."""
    scale = np.random.uniform(*scale_range)
    img_f = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_f * 255.0 / scale) * scale / 255.0
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return noisy


def shift_hue(img, max_shift=20):
    """Dịch Hue ngẫu nhiên (giả lập cân bằng trắng sai)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    shift = np.random.randint(-max_shift, max_shift + 1)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def shift_saturation(img, range_factor=(0.5, 1.5)):
    """Thay đổi độ bão hòa (giả lập màu bị nhạt hoặc quá rực)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = np.random.uniform(*range_factor)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def shift_brightness(img, range_factor=(0.6, 1.4)):
    """Thay đổi độ sáng (giả lập phơi sáng sai)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = np.random.uniform(*range_factor)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def shift_color_channels(img, max_shift=15):
    """Dịch từng kênh BGR độc lập (giả lập color cast)."""
    result = img.astype(np.float32)
    for ch in range(3):
        shift = np.random.uniform(-max_shift, max_shift)
        result[:, :, ch] = np.clip(result[:, :, ch] + shift, 0, 255)
    return result.astype(np.uint8)


def add_compression_artifact(img, quality_range=(15, 40)):
    """Thêm artifact nén JPEG (giả lập video chất lượng thấp)."""
    quality = np.random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def degrade_frame(clean_frame, noise_level='medium'):
    """
    Áp dụng chuỗi biến đổi ngẫu nhiên lên frame sạch để tạo frame "hỏng".
    Model sẽ học cách đảo ngược tất cả biến đổi này.

    noise_level: 'light', 'medium', 'heavy'
    """
    degraded = clean_frame.copy()

    # Cấu hình theo mức độ
    configs = {
        'light':  {'sigma': (3, 15),  'hue': 8,  'sat': (0.8, 1.2), 'bright': (0.85, 1.15), 'color': 8},
        'medium': {'sigma': (5, 35),  'hue': 15, 'sat': (0.6, 1.4), 'bright': (0.7, 1.3),  'color': 12},
        'heavy':  {'sigma': (15, 50), 'hue': 25, 'sat': (0.4, 1.6), 'bright': (0.5, 1.5),  'color': 20},
    }
    cfg = configs.get(noise_level, configs['medium'])

    # 1. Nhiễu (luôn áp dụng — đây là tính năng cốt lõi)
    if np.random.random() < 0.7:
        degraded = add_gaussian_noise(degraded, sigma_range=cfg['sigma'])
    else:
        degraded = add_poisson_noise(degraded)

    # 2. Sai lệch màu (áp dụng ngẫu nhiên 1-3 kiểu)
    if np.random.random() < 0.6:
        degraded = shift_hue(degraded, max_shift=cfg['hue'])

    if np.random.random() < 0.5:
        degraded = shift_saturation(degraded, range_factor=cfg['sat'])

    if np.random.random() < 0.5:
        degraded = shift_brightness(degraded, range_factor=cfg['bright'])

    if np.random.random() < 0.4:
        degraded = shift_color_channels(degraded, max_shift=cfg['color'])

    # 3. Artifact nén (ngẫu nhiên)
    if np.random.random() < 0.3:
        degraded = add_compression_artifact(degraded)

    return degraded
