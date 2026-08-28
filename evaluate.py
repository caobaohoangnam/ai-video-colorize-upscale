import cv2
import numpy as np
import time
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def _delta_e_mean(frame_o, frame_r):
    """Tính Delta-E (CIE76) trung bình giữa 2 frame — thước đo chuẩn cho color correction."""
    lab_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab_o - lab_r
    delta_e = np.sqrt(np.sum(diff ** 2, axis=2))  # (H, W)
    return float(np.mean(delta_e))


def evaluate_video(original_path, restored_path, report_name="evaluation_report.csv"):
    cap_orig = cv2.VideoCapture(original_path)
    cap_rest = cv2.VideoCapture(restored_path)

    # Kiểm tra file
    if not cap_orig.isOpened() or not cap_rest.isOpened():
        print("Lỗi: Không thể mở một trong hai file video.")
        return

    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    total_rest = int(cap_rest.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_orig != total_rest:
        print(f"Cảnh báo: Số khung hình không khớp — gốc={total_orig}, phục hồi={total_rest}. "
              f"Sẽ đánh giá đến khung hình ngắn hơn.")

    psnr_values = []
    ssim_values = []
    delta_e_values = []
    per_frame_rows = []
    frame_count = 0

    print(f"Đang bắt đầu đánh giá: {os.path.basename(restored_path)}...")
    start_time = time.time()

    while True:
        ret_o, frame_o = cap_orig.read()
        ret_r, frame_r = cap_rest.read()

        if not ret_o or not ret_r:
            break

        # Đảm bảo cùng kích thước để so sánh
        if frame_o.shape != frame_r.shape:
            frame_r = cv2.resize(frame_r, (frame_o.shape[1], frame_o.shape[0]))

        # --- PSNR (toàn màu BGR) ---
        psnr_val = psnr(frame_o, frame_r, data_range=255)
        # Bỏ qua frame identical (PSNR = inf) để tránh làm hỏng trung bình
        if np.isinf(psnr_val):
            psnr_val = None

        # --- SSIM (ảnh xám, data_range rõ ràng) ---
        gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_o, gray_r, data_range=255)

        # --- Delta-E (đo chất lượng màu sắc trong không gian LAB) ---
        de_val = _delta_e_mean(frame_o, frame_r)

        if psnr_val is not None:
            psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        delta_e_values.append(de_val)
        frame_count += 1

        per_frame_rows.append([frame_count,
                                f"{psnr_val:.2f}" if psnr_val is not None else "inf",
                                f"{ssim_val:.4f}",
                                f"{de_val:.4f}"])

        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} khung hình...")

    elapsed = time.time() - start_time
    fps_throughput = frame_count / elapsed if elapsed > 0 else 0

    # Tính thống kê
    avg_psnr = np.mean(psnr_values) if psnr_values else float('nan')
    std_psnr = np.std(psnr_values) if psnr_values else float('nan')
    min_psnr = np.min(psnr_values) if psnr_values else float('nan')

    avg_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    min_ssim = np.min(ssim_values)

    avg_de = np.mean(delta_e_values)
    std_de = np.std(delta_e_values)
    max_de = np.max(delta_e_values)  # cao = lỗi màu lớn

    # Ghi kết quả ra CSV
    with open(report_name, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Tóm tắt
        writer.writerow(["=== TÓM TẮT ==="])
        writer.writerow(["Metric", "Average", "Std Dev", "Min/Max"])
        writer.writerow(["PSNR (dB) ↑ cao hơn là tốt hơn",
                         f"{avg_psnr:.2f}", f"{std_psnr:.2f}", f"min={min_psnr:.2f}"])
        writer.writerow(["SSIM ↑ gần 1 là tốt hơn",
                         f"{avg_ssim:.4f}", f"{std_ssim:.4f}", f"min={min_ssim:.4f}"])
        writer.writerow(["Delta-E ↓ thấp hơn là tốt hơn (màu sắc)",
                         f"{avg_de:.4f}", f"{std_de:.4f}", f"max={max_de:.4f}"])
        writer.writerow(["Tổng số khung hình", frame_count])
        writer.writerow(["Tốc độ xử lý (FPS)", f"{fps_throughput:.1f}"])
        writer.writerow([])

        # Dữ liệu từng frame
        writer.writerow(["=== DỮ LIỆU TỪNG FRAME ==="])
        writer.writerow(["Frame", "PSNR (dB)", "SSIM", "Delta-E"])
        writer.writerows(per_frame_rows)

    print("-" * 40)
    print(f"KẾT QUẢ ĐÁNH GIÁ ({frame_count} khung hình):")
    print(f"  PSNR trung bình : {avg_psnr:.2f} dB  (±{std_psnr:.2f})")
    print(f"  SSIM trung bình : {avg_ssim:.4f}      (±{std_ssim:.4f})")
    print(f"  Delta-E TB      : {avg_de:.4f}      (±{std_de:.4f})  ← chỉ số màu sắc")
    print(f"  Tốc độ xử lý   : {fps_throughput:.1f} FPS")
    print(f"  Báo cáo lưu tại : {report_name}")
    print("-" * 40)

    # --- Vẽ biểu đồ ---
    frames_x = list(range(1, frame_count + 1))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Evaluation Results — {os.path.basename(restored_path)}',
                 fontsize=14, fontweight='bold')

    # PSNR per frame
    ax = axes[0, 0]
    psnr_plot = [v if v is not None else float('nan') for v in
                 [row[1] for row in per_frame_rows]]
    psnr_float = [float(v) if v != 'inf' else float('nan') for v in psnr_plot]
    ax.plot(frames_x, psnr_float, color='steelblue', linewidth=0.8, label='PSNR')
    ax.axhline(avg_psnr, color='orange', linewidth=2, linestyle='--',
               label=f'avg={avg_psnr:.2f} dB')
    ax.set_title('PSNR per Frame (dB) ↑')
    ax.set_xlabel('Frame')
    ax.set_ylabel('dB')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSIM per frame
    ax = axes[0, 1]
    ax.plot(frames_x, ssim_values, color='green', linewidth=0.8, label='SSIM')
    ax.axhline(avg_ssim, color='orange', linewidth=2, linestyle='--',
               label=f'avg={avg_ssim:.4f}')
    ax.set_title('SSIM per Frame ↑  (closer to 1 = better)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('SSIM')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Delta-E per frame
    ax = axes[1, 0]
    ax.plot(frames_x, delta_e_values, color='tomato', linewidth=0.8, label='Delta-E')
    ax.axhline(avg_de, color='orange', linewidth=2, linestyle='--',
               label=f'avg={avg_de:.4f}')
    ax.set_title('Delta-E per Frame ↓  (lower = better color)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Delta-E')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary bar chart
    ax = axes[1, 1]
    metrics   = ['PSNR (dB)', 'SSIM ×100', 'Delta-E']
    averages  = [avg_psnr, avg_ssim * 100, avg_de]
    colors    = ['steelblue', 'green', 'tomato']
    bars = ax.bar(metrics, averages, color=colors, width=0.5)
    for bar, val in zip(bars, averages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Summary (Average Metrics)')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = report_name.replace('.csv', '_charts.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Biểu đồ đã lưu tại: {chart_path}")

    cap_orig.release()
    cap_rest.release()


if __name__ == "__main__":
    import os

    input_dir  = "data/input"
    output_dir = "data/output"

    videos = [f for f in os.listdir(output_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]

    if not videos:
        print(f"Không tìm thấy video nào trong {output_dir}/")
        exit(1)

    print("Video có trong data/output:")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {v}")

    choice = input("\nNhập tên video (hoặc số thứ tự): ").strip()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            video_name = videos[idx]
        else:
            print("Số thứ tự không hợp lệ.")
            exit(1)
    else:
        video_name = choice
        if not os.path.exists(os.path.join(output_dir, video_name)):
            print(f"Không tìm thấy file: {video_name}")
            exit(1)

    ORIGINAL = os.path.join(input_dir, video_name)
    RESTORED = os.path.join(output_dir, video_name)

    if not os.path.exists(ORIGINAL):
        print(f"Không tìm thấy file gốc tương ứng: {ORIGINAL}")
        exit(1)

    evaluate_video(ORIGINAL, RESTORED)
