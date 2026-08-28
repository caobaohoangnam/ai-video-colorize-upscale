import torch
import cv2
import os
import yaml

from tqdm import tqdm
from models.network import VideoRestorationNet
from utils.video_utils import get_video_info, frame_to_tensor, tensor_to_frame, transfer_audio, get_scene_cuts
from utils.upscale_utils import get_upscaler, upscale_frame
from utils.color_utils import apply_color_correction, TemporalColorSmoother
from utils.colorize_utils import load_colorizer, detect_bw_video, colorize_frame
from utils.scene_utils import detect_scene_type, apply_scene_color_hints, apply_animation_style, detect_animation_video


def run_restoration(config_path):
    # 1. Tải cấu hình
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Thiết lập thiết bị
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Đang chạy trên: {torch.cuda.get_device_name(0)}")
    else:
        print("Đang chạy trên CPU (sẽ chậm hơn nhiều).")

    use_fp16 = device.type == 'cuda'

    # 3. Khởi tạo bộ upscale
    upscale_factor = config['processing'].get('upscale_factor', 2)
    target_h_cfg = config['processing'].get('target_resolution', 0)
    # Nếu có target resolution, dùng outscale=4 để Real-ESRGAN cho chất lượng tối đa
    # rồi resize về đúng target sau. Giúp 144p→1440p không bị mờ.
    esrgan_outscale = 4 if target_h_cfg else upscale_factor
    upsampler = get_upscaler(config['paths']['checkpoint_upscale'], device, outscale=esrgan_outscale)

    # Bảng độ phân giải chuẩn (chiều cao → tên)
    RESOLUTIONS = {
        144:  "144p",
        240:  "240p",
        360:  "360p",
        480:  "480p",
        720:  "720p (HD)",
        1080: "1080p (Full HD)",
        1440: "1440p (2K)",
        2160: "2160p (4K)",
    }

    # 4. Khởi tạo mô hình khử nhiễu và nạp vào GPU
    t_window = config['model_settings']['temporal_window']
    model = VideoRestorationNet(temporal_window=t_window).to(device)

    if os.path.exists(config['paths']['checkpoint_denoise']):
        state = torch.load(config['paths']['checkpoint_denoise'], map_location=device)
        # Hỗ trợ cả checkpoint từ train.py (dict) và file .pth thuần
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        model.eval()
        if use_fp16:
            model.half()
        print("Đã nạp trọng số mô hình khử nhiễu thành công.")
    else:
        print("Cảnh báo: Không tìm thấy file weights. Mô hình sẽ chạy với tham số ngẫu nhiên (chỉ để test code).")
        model.eval()
        if use_fp16:
            model.half()

    # 5. Chuẩn bị Video
    # Nếu chỉ nhập tên file (không có thư mục), tự động tìm trong data/input/
    input_video = config['paths']['input_video']
    if not os.path.dirname(input_video):
        input_video = os.path.join('data/input', input_video)
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Không tìm thấy video: {input_video}")
    config['paths']['input_video'] = input_video

    info = get_video_info(input_video)
    cap = cv2.VideoCapture(input_video)

    # Tính kích thước resize trước khi khử nhiễu (tránh OOM trên GPU 4GB)
    max_denoise_size = config['processing'].get('max_denoise_size', 1280)
    orig_w, orig_h = info['width'], info['height']
    scale_factor = min(max_denoise_size / max(orig_w, orig_h), 1.0)
    denoise_w = int(orig_w * scale_factor / 2) * 2
    denoise_h = int(orig_h * scale_factor / 2) * 2
    needs_resize = (denoise_w != orig_w or denoise_h != orig_h)
    if needs_resize:
        print(f"Video {orig_w}×{orig_h} quá lớn — resize xuống {denoise_w}×{denoise_h} trước khi khử nhiễu.")

    target_h = config['processing'].get('target_resolution', 0)
    if target_h and target_h in RESOLUTIONS:
        # Tính chiều rộng giữ tỉ lệ khung hình
        aspect = orig_w / orig_h
        target_w = int(target_h * aspect / 2) * 2  # Đảm bảo chẵn
        out_width, out_height = target_w, target_h
        res_name = RESOLUTIONS[target_h]
        print(f"Độ phân giải đầu ra: {out_width}×{out_height} ({res_name})")
    else:
        out_width  = orig_w * upscale_factor
        out_height = orig_h * upscale_factor
        print(f"Độ phân giải đầu ra: {out_width}×{out_height} ({upscale_factor}x upscale)")

    temp_output = os.path.join(config['paths']['output_dir'], "temp_no_audio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*config['processing']['save_format'])
    out = cv2.VideoWriter(temp_output, fourcc, info['fps'], (out_width, out_height))

    # 6. Phát hiện điểm chuyển cảnh
    cut_frames = get_scene_cuts(config['paths']['input_video'], threshold=config['processing']['scene_threshold'])

    # 7. Giới hạn số frame nếu đang test
    frame_limit = config['processing'].get('frame_limit', 0)
    total_frames = info['total_frames']
    if frame_limit > 0:
        total_frames = min(total_frames, frame_limit)
        print(f"Chế độ test: chỉ xử lý {total_frames} khung hình đầu.")

    # 8. Cấu hình tô màu đen trắng
    color_cfg_bw = config.get('colorization', {})
    colorize_mode = color_cfg_bw.get('enabled', 'auto')
    colorizer = None

    if colorize_mode == 'auto':
        print("Đang kiểm tra video có phải hoạt hình không...")
        is_animation = detect_animation_video(cap)
        if is_animation:
            print("Phát hiện hoạt hình — bỏ qua tô màu để giữ phong cách nghệ thuật gốc.")
        else:
            print("Đang kiểm tra video có phải đen trắng không...")
            if detect_bw_video(cap):
                print("Phát hiện video đen trắng — sẽ tô màu tự động.")
                try:
                    colorizer = load_colorizer()
                except RuntimeError as e:
                    print(f"Cảnh báo: {e}\nBỏ qua bước tô màu, tiếp tục xử lý.")
            else:
                print("Video đã có màu — bỏ qua bước tô màu.")
    elif colorize_mode is True:
        print("Tô màu: BẬT (theo cấu hình)")
        try:
            colorizer = load_colorizer()
        except RuntimeError as e:
            print(f"Cảnh báo: {e}\nBỏ qua bước tô màu, tiếp tục xử lý.")
    else:
        print("Tô màu: TẮT")

    # 9. Cấu hình phân tích cảnh
    scene_cfg = config.get('scene_aware', {})
    use_scene_aware = scene_cfg.get('enabled', True)
    use_animation_style = scene_cfg.get('animation_style', True)
    if use_scene_aware:
        print("Phân tích cảnh: BẬT (đêm / cháy nổ / ngoài trời / hoạt hình)")

    # 10. Cấu hình làm mượt màu theo thời gian
    tc_cfg = config.get('temporal_consistency', {})
    use_temporal = tc_cfg.get('enabled', True)
    tc_alpha = tc_cfg.get('alpha', 0.75)
    color_smoother = TemporalColorSmoother(alpha=tc_alpha) if use_temporal else None
    if use_temporal:
        print(f"Làm mượt màu thời gian: BẬT (alpha={tc_alpha})")

    # 10. Cấu hình chỉnh màu hậu kỳ
    color_cfg = config.get('color_correction', {})
    use_color_correction = color_cfg.get('enabled', False)
    if use_color_correction:
        print("Chỉnh màu hậu kỳ: BẬT")

    # 10. Vòng lặp xử lý (Sliding Window)
    print(f"Bắt đầu xử lý {total_frames} khung hình...")

    frames_buffer = []

    def process_and_write(buffer):
        """Xử lý buffer qua model → tô màu → chỉnh màu → upscale → ghi file."""
        input_tensor = torch.cat(buffer, dim=1)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        res_frame = tensor_to_frame(output_tensor)

        # Khôi phục kích thước gốc nếu đã resize trước khi khử nhiễu
        if needs_resize:
            res_frame = cv2.resize(res_frame, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # Tô màu đen trắng (chạy trên CPU, không tốn VRAM)
        if colorizer is not None:
            res_frame = colorize_frame(colorizer, res_frame)

        # Phân tích cảnh + áp dụng color hints
        if use_scene_aware:
            scene = detect_scene_type(res_frame)
            res_frame = apply_scene_color_hints(res_frame, scene)
            if use_animation_style and scene['is_animation']:
                res_frame = apply_animation_style(res_frame)

        # Làm mượt màu theo thời gian (chống flickering)
        if color_smoother is not None:
            res_frame = color_smoother.smooth(res_frame)

        # Chỉnh màu hậu kỳ (chạy trên CPU, không tốn VRAM)
        if use_color_correction:
            res_frame = apply_color_correction(res_frame, color_cfg)

        frame_up = upscale_frame(upsampler, res_frame)

        # Resize về đúng độ phân giải mục tiêu nếu cần
        if frame_up.shape[1] != out_width or frame_up.shape[0] != out_height:
            interp = cv2.INTER_AREA if frame_up.shape[0] > out_height else cv2.INTER_LANCZOS4
            frame_up = cv2.resize(frame_up, (out_width, out_height), interpolation=interp)

        out.write(frame_up)

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # KIỂM TRA CHUYỂN CẢNH — xóa buffer để tránh màu cảnh cũ tràn sang cảnh mới
        if i in cut_frames:
            # Xử lý các frame còn lại trong buffer trước khi reset
            while len(frames_buffer) >= t_window:
                process_and_write(frames_buffer[:t_window])
                frames_buffer.pop(0)
            frames_buffer.clear()
            model.reset_state()
            if color_smoother is not None:
                color_smoother.reset()
            print(f"Phát hiện chuyển cảnh tại frame {i}, đã reset bộ nhớ.")

        if needs_resize:
            frame = cv2.resize(frame, (denoise_w, denoise_h), interpolation=cv2.INTER_AREA)
        img_t = frame_to_tensor(frame, device, use_fp16=use_fp16)
        frames_buffer.append(img_t)

        # Khi đủ số lượng khung hình cho cửa sổ thời gian
        if len(frames_buffer) == t_window:
            process_and_write(frames_buffer)
            frames_buffer.pop(0)

    # 11. Xử lý các frame CUỐI còn sót trong buffer
    #    Padding bằng cách lặp lại frame cuối để model vẫn nhận đủ t_window frame
    while len(frames_buffer) > 0 and len(frames_buffer) < t_window:
        frames_buffer.append(frames_buffer[-1])  # Lặp frame cuối
    if len(frames_buffer) == t_window:
        process_and_write(frames_buffer)

    # 12. Dọn dẹp và ghép âm thanh
    cap.release()
    out.release()

    input_filename = os.path.basename(config['paths']['input_video'])
    final_output = os.path.join(config['paths']['output_dir'], input_filename)
    transfer_audio(config['paths']['input_video'], temp_output, final_output)


if __name__ == "__main__":
    input_dir = "data/input"
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]

    if not videos:
        print(f"Không tìm thấy video nào trong {input_dir}/")
        exit(1)

    print("Video có trong data/input:")
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
        if not os.path.exists(os.path.join(input_dir, video_name)):
            print(f"Không tìm thấy file: {video_name}")
            exit(1)

    # Chọn độ phân giải đầu ra
    RESOLUTIONS = {
        1: (720,  "720p (HD)"),
        2: (1080, "1080p (Full HD)"),
        3: (1440, "1440p (2K)"),
        4: (2160, "2160p (4K)"),
    }
    print("\nChọn độ phân giải đầu ra:")
    for k, (h, name) in RESOLUTIONS.items():
        print(f"  [{k}] {name}")
    print("  [0] Dùng upscale_factor trong config.yaml")

    res_choice = input("Nhập số thứ tự: ").strip()
    target_resolution = 0
    if res_choice.isdigit() and int(res_choice) in RESOLUTIONS:
        target_resolution = RESOLUTIONS[int(res_choice)][0]
        print(f"Đã chọn: {RESOLUTIONS[int(res_choice)][1]}")

    # Ghi đè đường dẫn input_video trong config
    with open("configs/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths']['input_video'] = os.path.join(input_dir, video_name)
    if target_resolution:
        config['processing']['target_resolution'] = target_resolution

    # Chạy với config đã chỉnh
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
    yaml.dump(config, tmp, allow_unicode=True)
    tmp.close()
    run_restoration(tmp.name)
    os.unlink(tmp.name)
