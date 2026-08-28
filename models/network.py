import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Khối phần dư giúp mô hình học sâu hơn mà không bị mất chi tiết."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.conv(x)


class VideoRestorationNet(nn.Module):
    def __init__(self, input_channels=3, temporal_window=5):
        super(VideoRestorationNet, self).__init__()

        self.temporal_window = temporal_window
        # Tổng số kênh đầu vào = 3 (RGB) * số khung hình liên tiếp
        self.start_channels = input_channels * temporal_window

        # --- Encoder (Nén và trích xuất đặc trưng) ---
        # Dùng strided conv thay MaxPool để giữ lại chi tiết tốt hơn cho restoration
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.start_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.down1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.down2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)

        # --- Bottleneck (Lớp xử lý sâu nhất — tại 1/4 độ phân giải) ---
        self.bottleneck = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # --- Decoder (Khôi phục chi tiết và chỉnh màu) ---
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),  # Ghép với enc2 (Skip connection)
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),  # Ghép với enc1 (Skip connection)
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Trả về 3 kênh màu RGB
        )

        # Sigmoid đảm bảo đầu ra luôn nằm trong [0, 1], tránh artifact màu
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor có dạng (Batch, Kênh * Cửa sổ thời gian, H, W)
        Sử dụng residual learning: kết quả = frame trung tâm + correction từ mô hình.
        """
        # Trích xuất frame trung tâm làm cơ sở (residual learning)
        mid = self.temporal_window // 2
        center_frame = x[:, mid*3:(mid+1)*3, :, :]  # (B, 3, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        # Bottleneck
        b = self.bottleneck(self.down2(e2))

        # Decoder với Skip Connections
        # Dùng F.interpolate để khớp kích thước khi input có chiều lẻ
        u2 = self.up2(b)
        if u2.shape != e2.shape:
            u2 = u2[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape != e1.shape:
            u1 = u1[:, :, :e1.shape[2], :e1.shape[3]]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Residual learning: mô hình học phần chỉnh sửa, cộng vào frame gốc
        return self.output_act(center_frame + d1)

    def reset_state(self):
        """
        Reset trạng thái khi phát hiện chuyển cảnh.
        Mô hình feedforward này không có hidden state,
        nhưng method này được gọi từ main.py để xóa temporal buffer.
        Dự phòng nếu sau này thêm recurrent layers (ConvLSTM, etc.).
        """
        pass


# Test thử mô hình
if __name__ == "__main__":
    # Test với kích thước chẵn
    model = VideoRestorationNet(temporal_window=5)
    dummy_input = torch.randn(1, 15, 256, 256)  # 15 = 3 channels * 5 frames
    output = model(dummy_input)
    print(f"Đầu vào: {dummy_input.shape}")
    print(f"Đầu ra: {output.shape}")  # Phải là (1, 3, 256, 256)

    # Test với kích thước LẺ (trước đây sẽ crash)
    dummy_odd = torch.randn(1, 15, 271, 481)
    output_odd = model(dummy_odd)
    print(f"Đầu vào lẻ: {dummy_odd.shape}")
    print(f"Đầu ra lẻ: {output_odd.shape}")  # Phải là (1, 3, 271, 481)

    # Kiểm tra đầu ra nằm trong [0, 1]
    print(f"Giá trị min: {output.min():.4f}, max: {output.max():.4f}")  # Phải >= 0 và <= 1
