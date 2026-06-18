# CLAUDE.md

Rules đầy đủ của project nằm ở **[AGENTS.md](AGENTS.md)** (dùng chung cho mọi AI agent). Import vào đây:

@AGENTS.md

## Ghi chú riêng cho Claude Code

- Khi sửa notebook (`.ipynb`): thao tác trên JSON bằng script Python (json.load → sửa `cell['source']` → json.dump), đừng sửa tay chuỗi JSON đã escape.
- Máy local là **Windows 11** (PowerShell + Git Bash), Python 3.14 (KHÔNG có TensorFlow). Việc cần TF (train/convert) chạy trên **server `har_fall`**. Kiểm tra/parse `.tflite`, gzip, dựng lại file từ `.cc` thì làm được bằng Python thuần local.
- Khi đề xuất kiến trúc/đánh giá, bám mục §4 (gotchas) của AGENTS.md — nhất là ESP-NN accel list và ràng buộc Keras 2 cho LSTM.
