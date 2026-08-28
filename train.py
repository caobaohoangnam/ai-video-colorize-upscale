"""
Training script cho VideoRestorationNet.

Tối ưu cho 4GB VRAM:
- Patch size 256x256
- Batch size 2
- FP16 mixed precision (AMP)
- Gradient accumulation nếu cần batch lớn hơn

Loss = L1 (chi tiết) + SSIM (cấu trúc) + LAB color (chỉnh màu)

Cách dùng:
    python train.py
    python train.py --config configs/config.yaml
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import yaml

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from models.network import VideoRestorationNet
from utils.dataset import VideoFrameDataset


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class SSIMLoss(nn.Module):
    """SSIM loss — đo sự tương đồng cấu trúc, bổ sung cho L1."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        # Tính trung bình cục bộ bằng average pooling
        pad = self.window_size // 2
        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=pad)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=pad)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(pred ** 2, self.window_size, stride=1, padding=pad) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, self.window_size, stride=1, padding=pad) - mu_target_sq
        sigma_cross = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=pad) - mu_cross

        ssim_map = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1.0 - ssim_map.mean()  # 1 - SSIM để biến thành loss (nhỏ hơn = tốt hơn)


class LABColorLoss(nn.Module):
    """
    Loss trong không gian LAB — đo sai lệch màu sắc.
    Chuyển đổi RGB → LAB trên GPU, tính L1 trên kênh A,B (màu sắc).
    Kênh L (độ sáng) đã được L1 loss chính lo.
    """

    def forward(self, pred, target):
        # pred, target: (B, 3, H, W) trong [0, 1], RGB order
        # Chuyển sang không gian gần LAB bằng cách tách luminance vs chrominance
        # (Xấp xỉ nhanh trên GPU thay vì dùng cv2 trên CPU)
        pred_r, pred_g, pred_b = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        target_r, target_g, target_b = target[:, 0:1], target[:, 1:2], target[:, 2:3]

        # Chrominance channels (Cb, Cr kiểu YCbCr — tương tự ý nghĩa kênh A,B trong LAB)
        pred_cb = -0.169 * pred_r - 0.331 * pred_g + 0.500 * pred_b
        pred_cr = 0.500 * pred_r - 0.419 * pred_g - 0.081 * pred_b

        target_cb = -0.169 * target_r - 0.331 * target_g + 0.500 * target_b
        target_cr = 0.500 * target_r - 0.419 * target_g - 0.081 * target_b

        color_loss = F.l1_loss(pred_cb, target_cb) + F.l1_loss(pred_cr, target_cr)
        return color_loss


class CombinedLoss(nn.Module):
    """Kết hợp 3 loss: L1 (chi tiết) + SSIM (cấu trúc) + Color (màu sắc)."""

    def __init__(self, l1_weight=1.0, ssim_weight=0.5, color_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.color_loss = LABColorLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        color = self.color_loss(pred, target)

        total = self.l1_weight * l1 + self.ssim_weight * ssim + self.color_weight * color
        return total, {'l1': l1.item(), 'ssim': ssim.item(), 'color': color.item()}


# ============================================================
# TRAINING LOOP
# ============================================================

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_cfg = config['training']
    t_window = config['model_settings']['temporal_window']

    # --- Dataset & Dataloader ---
    dataset = VideoFrameDataset(
        video_dir=train_cfg['clean_video_dir'],
        patch_size=train_cfg['patch_size'],
        temporal_window=t_window,
        noise_level=train_cfg['noise_level'],
        max_frames_per_video=train_cfg.get('max_frames_per_video', 500)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )

    # --- Model ---
    model = VideoRestorationNet(temporal_window=t_window).to(device)

    # Nạp checkpoint nếu tiếp tục train
    start_epoch = 0
    checkpoint_path = config['paths']['checkpoint_denoise']
    if os.path.exists(checkpoint_path) and train_cfg.get('resume', False):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
            start_epoch = state.get('epoch', 0)
            print(f"Tiếp tục training từ epoch {start_epoch}")
        else:
            model.load_state_dict(state)
            print("Đã nạp trọng số model (không có thông tin epoch).")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 1e-4)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'],
        eta_min=train_cfg['learning_rate'] * 0.01
    )

    # --- Loss ---
    criterion = CombinedLoss(
        l1_weight=train_cfg.get('l1_weight', 1.0),
        ssim_weight=train_cfg.get('ssim_weight', 0.5),
        color_weight=train_cfg.get('color_weight', 0.3)
    )

    # --- Mixed Precision (tiết kiệm VRAM ~40%) ---
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    grad_accum_steps = train_cfg.get('grad_accumulation', 1)

    # --- Training ---
    save_dir = os.path.dirname(checkpoint_path)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    history = {'total': [], 'l1': [], 'ssim': [], 'color': []}

    print(f"\nBắt đầu training: {train_cfg['epochs']} epochs, "
          f"batch={train_cfg['batch_size']}, patch={train_cfg['patch_size']}x{train_cfg['patch_size']}")
    print(f"Gradient accumulation: {grad_accum_steps} steps")
    print(f"Effective batch size: {train_cfg['batch_size'] * grad_accum_steps}\n")

    for epoch in range(start_epoch, train_cfg['epochs']):
        model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'ssim': 0, 'color': 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Mixed precision forward
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss, loss_dict = criterion(outputs, targets)
                loss = loss / grad_accum_steps

            # Backward
            scaler.scale(loss).backward()

            # Cập nhật weights sau mỗi grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Ghi nhận loss
            epoch_losses['total'] += loss.item() * grad_accum_steps
            epoch_losses['l1'] += loss_dict['l1']
            epoch_losses['ssim'] += loss_dict['ssim']
            epoch_losses['color'] += loss_dict['color']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item() * grad_accum_steps:.4f}",
                'l1': f"{loss_dict['l1']:.4f}",
                'color': f"{loss_dict['color']:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        scheduler.step()

        # Tính trung bình epoch
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        print(f"  → Epoch {epoch+1}: "
              f"total={epoch_losses['total']:.4f}, "
              f"l1={epoch_losses['l1']:.4f}, "
              f"ssim={epoch_losses['ssim']:.4f}, "
              f"color={epoch_losses['color']:.4f}")
        for k in history:
            history[k].append(epoch_losses[k])

        # Lưu checkpoint
        save_every = train_cfg.get('save_every', 5)
        if (epoch + 1) % save_every == 0 or epoch_losses['total'] < best_loss:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'loss': epoch_losses['total']
            }
            torch.save(state, checkpoint_path)
            print(f"  Đã lưu checkpoint tại epoch {epoch+1}")

            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                best_path = checkpoint_path.replace('.pth', '_best.pth')
                torch.save(state, best_path)
                print(f"  Đã lưu best model (loss={best_loss:.4f})")

    print(f"\nTraining hoàn tất! Best loss: {best_loss:.4f}")
    print(f"Model lưu tại: {checkpoint_path}")

    # --- Vẽ biểu đồ Loss Curves ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(history['total']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Curves — VideoRestorationNet', fontsize=15, fontweight='bold')

    pairs = [('total', 'Total Loss', axes[0, 0]),
             ('l1',    'L1 Loss (Detail)',    axes[0, 1]),
             ('ssim',  'SSIM Loss (Structure)', axes[1, 0]),
             ('color', 'Color Loss', axes[1, 1])]

    for key, title, ax in pairs:
        vals = history[key]
        ax.plot(epochs_range, vals, color='steelblue', linewidth=1.2, label='loss')
        # Smoothed trend line
        if len(vals) >= 5:
            import numpy as np
            smoothed = np.convolve(vals, np.ones(5)/5, mode='valid')
            ax.plot(range(3, 3 + len(smoothed)), smoothed,
                    color='orange', linewidth=2, linestyle='--', label='smooth')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'training_loss_curves.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Biểu đồ loss đã lưu tại: {chart_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoRestorationNet")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train(config)
