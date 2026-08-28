import torch
import numpy as np
import cv2
from realesrgan import RealESRGANer


def _build_model(model_name):
    """
    Chọn đúng kiến trúc mạng dựa theo tên file trọng số.
    - realesr-general-x4v3  → SRVGGNetCompact (nhỏ, nhanh, phù hợp 4GB VRAM)
    - RealESRGAN_x4plus     → RRDBNet (nặng, chất lượng cao hơn, cần 8GB+ VRAM)
    """
    if 'x4v3' in model_name or 'compact' in model_name:
        # SRVGGNetCompact — model nhỏ gọn (~170KB), tối ưu cho VRAM thấp
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=32,
            upscale=4, act_type='prelu'
        )
        native_scale = 4
    elif 'x2plus' in model_name:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        native_scale = 2
    else:
        # Mặc định: RRDBNet x4 (model gốc nặng ~67MB)
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        native_scale = 4

    return model, native_scale


def get_upscaler(model_path, device, outscale=2, tile=512):
    """
    Khởi tạo bộ nâng cấp Real-ESRGAN.

    Args:
        model_path: Đường dẫn tới file .pth
        device: torch device (cuda/cpu)
        outscale: Hệ số phóng đại đầu ra (2 = 720p→1440p ≈ 2K)
        tile: Kích thước ô xử lý (pixel). Lớn hơn = ít artifact ở mép ô,
              nhưng tốn VRAM hơn. 512 là an toàn cho 4GB với SRVGGNetCompact.
    """
    import os
    model_name = os.path.basename(model_path)
    model, native_scale = _build_model(model_name)

    use_half = device.type == 'cuda'

    upsampler = RealESRGANer(
        scale=native_scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device
    )
    # Lưu outscale để dùng khi enhance
    upsampler._outscale = outscale
    return upsampler


def upscale_frame(upsampler, frame):
    """
    Nâng cấp 1 khung hình lên độ phân giải cao hơn.
    outscale được lấy từ cấu hình đã gán lúc khởi tạo.
    """
    outscale = getattr(upsampler, '_outscale', 2)
    output, _ = upsampler.enhance(frame, outscale=outscale)
    # Giải phóng VRAM sau mỗi frame upscale
    torch.cuda.empty_cache()
    return output
