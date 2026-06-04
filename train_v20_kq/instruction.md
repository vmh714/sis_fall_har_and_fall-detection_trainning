# Tài liệu Instruction: Phiên bản TCN_MCU_Classification

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động dựa trên dữ liệu cảm biến IMU (6 features).
- **Quy mô:** Hỗ trợ 5 lớp đầu ra (5 classes) thông qua tầng Dense với hàm kích hoạt Softmax.

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên tiền tố hoặc từ khóa trong tên file:
- **Fall:** Các file có tiền tố bắt đầu bằng 'F'.
- **Walk:** Tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run:** Tiền tố 'D03', 'D04'.
- **Idle:** Chứa từ khóa '_StandSit_' hoặc '_Lie_'.
- **Trans:** Chứa từ khóa '_Trans_'.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Mỗi mẫu (sample) là một chuỗi thời gian cố định với độ dài 200 bước (200 timesteps).
- **Số lượng đặc trưng:** Dữ liệu đầu vào bao gồm 6 features từ cảm biến IMU.
- **Phân chia dữ liệu:** Chia tập Train/Val/Test dựa trên định danh đối tượng (`subject_id`) để đảm bảo tính độc lập giữa các tập dữ liệu.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **TCN (Temporal Convolutional Network)** tối ưu cho MCU:
    - 2 chồng (stack) lớp `Conv1D` với các độ trễ (dilation rates) lần lượt là [1, 2, 4, 8].
    - Sử dụng các khối Residual (tương đương kiến trúc ResNet) để tránh triệt tiêu gradient.
    - `BatchNormalization` và `Dropout` (0.2) được áp dụng sau mỗi lớp Conv1D để ổn định huấn luyện.
- **Kết hợp đặc trưng:** Sử dụng đồng thời `GlobalAveragePooling1D` (GAP) và `GlobalMaxPooling1D` (GMP) để bắt được cả đặc trưng nền (Idle) và các xung đỉnh (Fall/Trans).
- **Kỹ thuật tối ưu:**
    - Sử dụng **Class Weights** (chiến lược 'balanced') để xử lý sự mất cân bằng giữa các lớp.
    - Optimizer: **Adam** (learning rate = 1e-3).
    - Loss Function: **Sparse Categorical Crossentropy**.