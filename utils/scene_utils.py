"""
Scene analysis module — phát hiện loại cảnh và áp dụng color hints tương ứng.

Các loại cảnh được hỗ trợ:
  - night      : cảnh tối, đêm → tông màu lạnh (xanh dương)
  - fire       : cháy nổ, vụ nổ → tông màu nóng (cam/đỏ)
  - outdoor    : ngoài trời → bầu trời xanh, cỏ xanh lá
  - animation  : hoạt hình → màu sắc sặc sỡ, phẳng, tương phản cao
  - normal     : cảnh thường → giữ nguyên

Tất cả chạy trên CPU, không tốn VRAM.
"""

import cv2
import numpy as np


# ============================================================
# VIDEO-LEVEL ANIMATION DETECTION
# ============================================================

def detect_animation_video(cap, sample_count=12):
    """
    Lấy mẫu nhiều frame để xác định video có phải hoạt hình không.
    Trả về True nếu phần lớn frame có đặc trưng hoạt hình.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_count)
    anim_count = 0

    for i in range(sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            continue
        scene = detect_scene_type(frame)
        if scene['is_animation']:
            anim_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return anim_count >= (sample_count // 3)  # 1/3 frame là hoạt hình = đủ để xác định


# ============================================================
# SCENE DETECTION
# ============================================================

def detect_scene_type(frame):
    """
    Phân tích frame và trả về dict mô tả loại cảnh.

    Args:
        frame: BGR uint8 frame (có thể là grayscale giả hoặc đã tô màu)
    Returns:
        dict với các key: type, is_night, is_fire, is_outdoor, is_animation, brightness
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    scene = {
        'type': 'normal',
        'is_night': False,
        'is_fire': False,
        'is_outdoor': False,
        'is_animation': False,
        'brightness': mean_brightness,
    }

    # --- Phát hiện cảnh đêm / tối ---
    if mean_brightness < 55:
        scene['type'] = 'night'
        scene['is_night'] = True
        return scene

    # --- Phát hiện cháy nổ (vùng sáng cực cao chiếm > 4% frame) ---
    bright_ratio = float(np.sum(gray > 220)) / gray.size
    if bright_ratio > 0.04 and mean_brightness > 70:
        scene['type'] = 'fire'
        scene['is_fire'] = True
        return scene

    # --- Phát hiện hoạt hình ---
    # Đặc trưng: mật độ cạnh cao + màu phẳng trong từng vùng (std thấp)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.sum(edges > 0)) / gray.size

    h, w = gray.shape[:2]
    block_stds = []
    bsize = 32
    for y in range(0, h - bsize, bsize):
        for x in range(0, w - bsize, bsize):
            block_stds.append(float(np.std(gray[y:y+bsize, x:x+bsize])))
    avg_std = float(np.mean(block_stds)) if block_stds else 50.0

    if edge_density > 0.06 and avg_std < 28:
        scene['type'] = 'animation'
        scene['is_animation'] = True
        return scene

    # --- Phát hiện ngoài trời (phần trên frame có độ sáng cao và ít cạnh) ---
    top_region = gray[:h // 3, :]
    top_brightness = float(np.mean(top_region))
    top_edges = cv2.Canny(top_region, 50, 150)
    top_edge_density = float(np.sum(top_edges > 0)) / top_region.size

    if top_brightness > 140 and top_edge_density < 0.04:
        scene['type'] = 'outdoor'
        scene['is_outdoor'] = True

    return scene


# ============================================================
# COLOR HINTS PER SCENE TYPE
# ============================================================

def apply_scene_color_hints(frame, scene):
    """
    Áp dụng điều chỉnh màu sắc dựa trên loại cảnh đã phát hiện.
    Hoạt động trong không gian LAB để tách độ sáng khỏi màu sắc.

    Args:
        frame: BGR uint8 frame
        scene: dict từ detect_scene_type()
    Returns:
        frame BGR uint8 đã được điều chỉnh màu
    """
    scene_type = scene['type']
    if scene_type == 'normal':
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    if scene_type == 'night':
        # Tông lạnh: xanh dương + hơi tối
        b -= 10          # Đẩy về phía xanh dương
        a -= 4           # Hơi xanh lá (khử ấm)
        l *= 0.92        # Tối hơn một chút

    elif scene_type == 'fire':
        # Tông nóng: cam/đỏ ở vùng sáng
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bright_mask = np.clip((gray - 160) / 60.0, 0, 1)  # Gradient mask
        a += bright_mask * 18   # Đỏ hơn ở vùng sáng
        b += bright_mask * 25   # Vàng/cam ở vùng sáng

    elif scene_type == 'outdoor':
        h = frame.shape[0]
        # Bầu trời (1/3 trên): đẩy xanh dương
        sky_weight = np.zeros_like(l)
        sky_weight[:h//3, :] = 1.0
        b += sky_weight * 12   # Xanh dương cho bầu trời

        # Phần dưới: đẩy xanh lá (cỏ, cây)
        ground_weight = np.zeros_like(l)
        ground_weight[h//2:, :] = 0.6
        a -= ground_weight * 6   # Xanh lá nhẹ

    elif scene_type == 'animation':
        # Màu sặc sỡ, phẳng: khuếch đại kênh màu A, B
        a_c = a - 128
        b_c = b - 128
        a = 128 + a_c * 1.45
        b = 128 + b_c * 1.45
        # Tăng tương phản L nhẹ
        l_mean = float(np.mean(l))
        l = np.clip((l - l_mean) * 1.12 + l_mean, 0, 255)

    lab[:, :, 0] = np.clip(l, 0, 255)
    lab[:, :, 1] = np.clip(a, 0, 255)
    lab[:, :, 2] = np.clip(b, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ============================================================
# ANIMATION-SPECIFIC POST-PROCESSING
# ============================================================

def apply_animation_style(frame):
    """
    Xử lý chuyên biệt cho hoạt hình sau khi tô màu:
      1. Làm phẳng màu trong từng vùng (bilateral filter)
      2. Giữ sắc nét cạnh viền
      3. Tăng độ bão hòa mạnh hơn mức thông thường

    Args:
        frame: BGR uint8 frame đã tô màu
    Returns:
        frame BGR uint8 với phong cách hoạt hình
    """
    # Bilateral filter: làm mịn màu trong vùng nhưng giữ cạnh sắc
    smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # Tăng độ bão hòa
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Blend cạnh từ frame gốc để giữ độ sắc nét viền
    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_orig, 80, 180)
    edge_mask = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    edge_3ch = np.stack([edge_mask] * 3, axis=2).astype(np.float32) / 255.0

    result = (saturated.astype(np.float32) * (1 - edge_3ch * 0.6) +
              frame.astype(np.float32) * (edge_3ch * 0.6))
    return np.clip(result, 0, 255).astype(np.uint8)
