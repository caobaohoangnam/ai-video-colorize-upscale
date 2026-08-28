import cv2
import torch
import numpy as np
import subprocess
import os

from scenedetect import detect, ContentDetector


# ============================================================
# I/O MANAGEMENT
# ============================================================

def get_video_info(input_path):
    """Lấy các thông số cơ bản của video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Không thể mở video tại: {input_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info


def transfer_audio(input_video, output_video_no_audio, final_output):
    """
    Sử dụng FFMPEG để sao chép âm thanh từ video gốc sang video đã xử lý.
    """
    print("Đang đồng bộ âm thanh...")
    command = [
        'ffmpeg', '-y',
        '-i', output_video_no_audio, # Video đã xử lý (không âm thanh)
        '-i', input_video,           # Video gốc (để lấy âm thanh)
        '-map', '0:v:0',             # Lấy hình ảnh từ file 1
        '-map', '1:a:0?',            # Lấy âm thanh từ file 2 (nếu có)
        '-c:v', 'copy',              # Giữ nguyên video đã render
        '-c:a', 'aac',               # Nén âm thanh sang chuẩn AAC
        '-shortest',                 # Cắt theo file ngắn hơn
        final_output
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Xóa file tạm không có âm thanh nếu thành công
        if os.path.exists(output_video_no_audio):
            os.remove(output_video_no_audio)
        print(f"Xử lý hoàn tất! File cuối cùng: {final_output}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi ghép âm thanh: {e}")
        if e.stderr:
            print(f"Chi tiết từ FFmpeg: {e.stderr.decode()}")


# ============================================================
# PREPROCESSING
# ============================================================

def frame_to_tensor(frame, device, use_fp16=True):
    """
    Chuyển đổi từ OpenCV Frame (BGR, [0, 255])
    sang PyTorch Tensor (RGB, [0, 1], chuẩn NCHW).

    use_fp16: Dùng float16 để tiết kiệm VRAM (khuyến nghị cho card 4GB).
    """
    # Chuyển BGR sang RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Chuyển sang mảng float và chuẩn hóa về [0, 1]
    img = img.astype(np.float32) / 255.0
    # Chuyển từ (H, W, C) sang (C, H, W)
    img = torch.from_numpy(img).permute(2, 0, 1)
    # Thêm chiều Batch (N, C, H, W) và đưa lên thiết bị (CPU/GPU)
    tensor = img.unsqueeze(0).to(device)
    # Chuyển sang float16 để giảm một nửa VRAM sử dụng
    if use_fp16 and device.type == 'cuda':
        tensor = tensor.half()
    return tensor


# ============================================================
# POST-PROCESSING
# ============================================================

def tensor_to_frame(tensor):
    """
    Chuyển đổi từ PyTorch Tensor ([0, 1])
    về lại OpenCV Frame (BGR, [0, 255]).
    """
    # Loại bỏ chiều Batch, detach trước rồi chuyển về CPU
    img = tensor.squeeze(0).detach().float().permute(1, 2, 0).cpu().numpy()
    # Đưa về dải [0, 255] và kiểu dữ liệu uint8
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    # Chuyển RGB về lại BGR cho OpenCV
    result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Giải phóng VRAM đã dùng cho tensor này
    del tensor
    torch.cuda.empty_cache()
    return result


# ============================================================
# TEMPORAL (Scene Detection)
# ============================================================

def get_scene_cuts(video_path, threshold=27.0):
    """
    Trả về tập hợp (set) các frame có sự thay đổi cảnh đột ngột.
    Dùng set thay vì list để tra cứu O(1) thay vì O(n) mỗi frame.
    Threshold càng thấp càng nhạy (hợp với phim hoạt hình).
    """
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    # Chuyển đổi kết quả sang set số thứ tự frame
    cut_frames = {scene[0].frame_num for scene in scene_list}
    return cut_frames
