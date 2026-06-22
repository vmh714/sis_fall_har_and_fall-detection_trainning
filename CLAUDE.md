# CLAUDE.md

Rules đầy đủ của project nằm ở **[AGENTS.md](AGENTS.md)** (dùng chung cho mọi AI agent). Import vào đây:

@AGENTS.md

## Ghi chú riêng cho Claude Code

- Khi sửa notebook (`.ipynb`): BẮT BUỘC sử dụng công cụ `manage_notebook_cells.py` có sẵn ở thư mục gốc. Tuyệt đối không sửa trực tiếp chuỗi JSON để tránh lỗi. Cách dùng: `python manage_notebook_cells.py --notebook <path> --search-string "<Từ_khoá_nhận_diện_cell>" --code-file <file_code_tam_thoi.py>`. Chi tiết xem tại `d:\datn\.agents\skills\notebook_modification\SKILL.md`.
- Máy local là **Windows 11** (PowerShell + Git Bash), Python 3.14 (KHÔNG có TensorFlow). Việc cần TF (train/convert) chạy trên **server `har_fall`**. Kiểm tra/parse `.tflite`, gzip, dựng lại file từ `.cc` thì làm được bằng Python thuần local.
- Khi đề xuất kiến trúc/đánh giá, bám mục §4 (gotchas) của AGENTS.md — nhất là ESP-NN accel list và ràng buộc Keras 2 cho LSTM.
