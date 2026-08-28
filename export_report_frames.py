"""
Xuất frame mẫu để làm báo cáo.

Tự động lấy frame từ video gốc và video đã xử lý,
tạo ảnh so sánh before/after và lưu vào thư mục report_frames/.

Cách dùng:
    python export_report_frames.py
"""

import cv2
import os
import numpy as np


OUTPUT_DIR = "report_frames"
INPUT_DIR  = "data/input"
PROCESSED_DIR = "data/output"

# Số frame muốn xuất
NUM_SAMPLES = 5


def pick_video(folder, label):
    videos = [f for f in os.listdir(folder)
              if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    if not videos:
        print(f"Không tìm thấy video trong {folder}/")
        return None

    print(f"\nChọn video {label}:")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {v}")

    choice = input("Nhập số thứ tự: ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            return os.path.join(folder, videos[idx])
    print("Lựa chọn không hợp lệ.")
    return None


def get_evenly_spaced_frames(cap, n):
    """Lấy n frame phân bố đều trong video."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total * i / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def add_label(frame, text, color=(255, 255, 255)):
    """Thêm nhãn chữ lên góc trên trái của frame."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 13 + 10, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def make_comparison(frame_orig, frame_proc, frame_idx):
    """Ghép 2 frame thành ảnh so sánh left/right."""
    # Resize về cùng chiều cao
    h = min(frame_orig.shape[0], frame_proc.shape[0], 540)
    def resize_h(f):
        ratio = h / f.shape[0]
        return cv2.resize(f, (int(f.shape[1] * ratio), h))

    orig_r = resize_h(frame_orig)
    proc_r = resize_h(frame_proc)

    # Thêm nhãn
    orig_r = add_label(orig_r, "ORIGINAL", color=(100, 100, 255))
    proc_r = add_label(proc_r, "PROCESSED (AI)", color=(100, 255, 100))

    # Thêm đường kẻ dọc phân cách
    divider = np.zeros((h, 4, 3), dtype=np.uint8)
    divider[:] = (200, 200, 200)

    combined = np.hstack([orig_r, divider, proc_r])

    # Thêm tiêu đề
    title_bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    title_text = f"Frame {frame_idx}  |  Before vs After AI Restoration"
    cv2.putText(title_bar, title_text, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

    return np.vstack([title_bar, combined])


def save_individual(frame, name):
    """Lưu frame đơn lẻ."""
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, frame)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Chọn video
    orig_path = pick_video(INPUT_DIR, "GỐC (original)")
    if not orig_path:
        return

    proc_path = pick_video(PROCESSED_DIR, "ĐÃ XỬ LÝ (processed)")
    if not proc_path:
        return

    cap_orig = cv2.VideoCapture(orig_path)
    cap_proc = cv2.VideoCapture(proc_path)

    if not cap_orig.isOpened() or not cap_proc.isOpened():
        print("Không thể mở video.")
        return

    print(f"\nĐang xuất {NUM_SAMPLES} frame mẫu...")

    frames_orig = get_evenly_spaced_frames(cap_orig, NUM_SAMPLES)
    frames_proc = get_evenly_spaced_frames(cap_proc, NUM_SAMPLES)

    saved = []

    for i, ((idx_o, fo), (idx_p, fp)) in enumerate(zip(frames_orig, frames_proc)):
        # 1. Ảnh so sánh before/after
        comparison = make_comparison(fo, fp, idx_o)
        p = save_individual(comparison, f"comparison_{i+1}_frame{idx_o}.png")
        saved.append(p)

        # 2. Frame gốc riêng lẻ
        p = save_individual(add_label(fo, "ORIGINAL"), f"original_{i+1}_frame{idx_o}.png")
        saved.append(p)

        # 3. Frame đã xử lý riêng lẻ
        p = save_individual(add_label(fp, "PROCESSED (AI)"), f"processed_{i+1}_frame{idx_o}.png")
        saved.append(p)

        print(f"  Frame {i+1}/{NUM_SAMPLES} đã lưu.")

    # Tạo ảnh tổng hợp tất cả comparison
    all_comparisons = []
    for i, ((idx_o, fo), (_, fp)) in enumerate(zip(frames_orig, frames_proc)):
        all_comparisons.append(make_comparison(fo, fp, idx_o))

    if all_comparisons:
        # Resize về cùng chiều rộng trước khi stack
        max_w = max(c.shape[1] for c in all_comparisons)
        resized = []
        for c in all_comparisons:
            if c.shape[1] != max_w:
                c = cv2.resize(c, (max_w, c.shape[0]))
            resized.append(c)

        separator = np.full((8, max_w, 3), 60, dtype=np.uint8)
        grid = resized[0]
        for c in resized[1:]:
            grid = np.vstack([grid, separator, c])

        grid_path = os.path.join(OUTPUT_DIR, "ALL_comparisons.png")
        cv2.imwrite(grid_path, grid)
        print(f"\nẢnh tổng hợp: {grid_path}")

    cap_orig.release()
    cap_proc.release()

    print(f"\nĐã lưu {len(saved)} ảnh vào thư mục: {OUTPUT_DIR}/")
    print("Các file dành cho báo cáo:")
    print("  - comparison_*.png     → ảnh so sánh before/after (dùng cho Step 3, 10)")
    print("  - original_*.png       → frame gốc riêng lẻ")
    print("  - processed_*.png      → frame sau AI riêng lẻ")
    print("  - ALL_comparisons.png  → tổng hợp tất cả (dùng cho poster/slide)")


if __name__ == "__main__":
    main()
