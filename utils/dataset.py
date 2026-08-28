"""
Dataset cho training VideoRestorationNet.
Đọc video sạch → trích xuất chuỗi frame liên tiếp (temporal window)
→ áp dụng degradation ngẫu nhiên → trả về cặp (degraded_window, clean_center).
"""

import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.augment import degrade_frame


class VideoFrameDataset(Dataset):
    """
    Dataset tạo cặp (input, target) cho training.

    Cấu trúc thư mục:
        data/train/clean/   ← Đặt video chất lượng cao ở đây
        data/train/frames/  ← Sẽ tự động trích xuất frame vào đây

    Input:  t_window frame bị degraded, ghép theo channel → (3*t_window, H, W)
    Target: frame trung tâm sạch → (3, H, W)
    """

    def __init__(self, video_dir, patch_size=256, temporal_window=5,
                 noise_level='medium', max_frames_per_video=500):
        """
        Args:
            video_dir: Thư mục chứa video sạch (.mp4, .mkv, .avi)
            patch_size: Kích thước crop ngẫu nhiên (nhỏ hơn = tiết kiệm VRAM)
            temporal_window: Số frame liên tiếp model nhìn cùng lúc
            noise_level: 'light', 'medium', 'heavy'
            max_frames_per_video: Giới hạn frame mỗi video để tiết kiệm RAM
        """
        self.patch_size = patch_size
        self.temporal_window = temporal_window
        self.noise_level = noise_level
        self.mid = temporal_window // 2

        # Tìm tất cả video sạch
        video_exts = ('*.mp4', '*.mkv', '*.avi', '*.mov')
        video_paths = []
        for ext in video_exts:
            video_paths.extend(glob.glob(os.path.join(video_dir, ext)))

        if not video_paths:
            raise FileNotFoundError(f"Không tìm thấy video nào trong: {video_dir}")

        # Trích xuất frame từ tất cả video
        self.frame_sequences = []  # List of lists — mỗi video là 1 list frame paths
        frame_dir = os.path.join(os.path.dirname(video_dir), "frames")
        os.makedirs(frame_dir, exist_ok=True)

        for vpath in video_paths:
            vname = os.path.splitext(os.path.basename(vpath))[0].strip()
            # Xóa ký tự không hợp lệ trên Windows
            vname = "".join(c for c in vname if c not in r'\/:*?"<>|')
            vframe_dir = os.path.join(frame_dir, vname)

            # Chỉ trích xuất nếu chưa có
            if not os.path.exists(vframe_dir) or len(os.listdir(vframe_dir)) == 0:
                os.makedirs(vframe_dir, exist_ok=True)
                self._extract_frames(vpath, vframe_dir, max_frames_per_video)

            # Đọc danh sách frame đã trích xuất
            frames = sorted(glob.glob(os.path.join(vframe_dir, "*.png")))
            if len(frames) >= temporal_window:
                self.frame_sequences.append(frames)

        # Tính tổng số sample = tổng (len(seq) - t_window + 1) cho mỗi video
        self.samples = []
        for seq_idx, seq in enumerate(self.frame_sequences):
            for start in range(len(seq) - temporal_window + 1):
                self.samples.append((seq_idx, start))

        print(f"Dataset: {len(video_paths)} video, "
              f"{sum(len(s) for s in self.frame_sequences)} frame, "
              f"{len(self.samples)} sample windows")

    def _extract_frames(self, video_path, output_dir, max_frames):
        """Trích xuất frame từ video thành file PNG."""
        cap = cv2.VideoCapture(video_path)
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(output_dir, f"{count:06d}.png"), frame)
            count += 1
        cap.release()
        print(f"  Trích xuất {count} frame từ {os.path.basename(video_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, start = self.samples[idx]
        seq = self.frame_sequences[seq_idx]

        # Đọc t_window frame liên tiếp
        frames = []
        for i in range(self.temporal_window):
            frame = cv2.imread(seq[start + i])
            frames.append(frame)

        # Frame trung tâm = ground truth
        clean_center = frames[self.mid].copy()

        # Crop ngẫu nhiên (cùng vị trí cho tất cả frame)
        h, w = frames[0].shape[:2]
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            frames = [f[top:top+self.patch_size, left:left+self.patch_size] for f in frames]
            clean_center = clean_center[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize nếu frame nhỏ hơn patch_size
            frames = [cv2.resize(f, (self.patch_size, self.patch_size)) for f in frames]
            clean_center = cv2.resize(clean_center, (self.patch_size, self.patch_size))

        # Áp dụng degradation CÙNG MỨC cho tất cả frame trong window
        # (để giữ sự nhất quán thời gian — giống video thật)
        degraded_frames = [degrade_frame(f, self.noise_level) for f in frames]

        # Chuyển sang tensor
        # Input: ghép t_window degraded frames → (3*t_window, H, W)
        input_tensors = []
        for df in degraded_frames:
            t = cv2.cvtColor(df, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(t).permute(2, 0, 1)  # (3, H, W)
            input_tensors.append(t)
        input_tensor = torch.cat(input_tensors, dim=0)  # (3*t_window, H, W)

        # Target: frame trung tâm sạch → (3, H, W)
        target = cv2.cvtColor(clean_center, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target_tensor = torch.from_numpy(target).permute(2, 0, 1)  # (3, H, W)

        return input_tensor, target_tensor
