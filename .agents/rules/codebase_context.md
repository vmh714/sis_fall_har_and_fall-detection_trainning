---
trigger: always_on
glob: "*"
description: "Tự động nạp kiến trúc codebase vào context trước khi làm task — tránh grep/scan lại từ đầu. Bắt buộc cập nhật sau khi sửa codebase."
---

# Quy Tắc Ngữ Cảnh Codebase (Codebase Context)

## 1. Đọc trước khi làm

Trước khi thực hiện bất kỳ task nào liên quan đến backend, frontend, hoặc firmware, BẮT BUỘC đọc các file sau (theo vai trò):

| Vai trò | File cần đọc |
|---------|-------------|
| **Mọi task** | `datn-agent-skills/project_setup/architecture/overview.md` và `datn-agent-skills/project_setup/protocol.md` |
| **Backend** | `datn-agent-skills/project_setup/architecture/backend.md` và schema tại `backend/app/schemas/` |
| **Frontend** | `datn-agent-skills/project_setup/architecture/frontend.md` |
| **Firmware** | `datn-agent-skills/project_setup/architecture/firmware.md` |

> Mục đích: Biết ngay file nào ở đâu, endpoint nào tồn tại, luồng MQTT ra sao, schema thế nào — không cần Grep/Glob/Read lại toàn bộ codebase. Bắt buộc đọc ĐẦY ĐỦ các file trên trước khi thảo luận/code chức năng.

## 2. Cập nhật sau khi sửa codebase

Sau khi hoàn thành bất kỳ thay đổi nào ảnh hưởng đến kiến trúc, BẮT BUỘC cập nhật file tương ứng trong `project_setup/architecture/`:

| Loại thay đổi | File cần cập nhật |
|---------------|------------------|
| Thêm/xóa/sửa API endpoint | `backend.md` → mục API Endpoints |
| Thêm/xóa model SQLAlchemy | `backend.md` → mục PostgreSQL Schema |
| Thêm Alembic migration mới | `backend.md` → mục Migrations |
| Thêm InfluxDB measurement/field | `backend.md` → mục InfluxDB Measurements |
| Thêm/xóa page Next.js | `frontend.md` → mục Pages |
| Thêm/xóa component quan trọng | `frontend.md` → mục Components |
| Thêm/xóa store Zustand | `frontend.md` → mục Zustand Stores |
| Thay đổi MQTT topic | `overview.md` → mục MQTT Topics + file liên quan |
| Thêm package/dependency lớn | file liên quan → mục Tech Stack |
| Thay đổi luồng dữ liệu chính | `overview.md` + file liên quan |

## 3. Định dạng cập nhật

- Luôn cập nhật dòng `> **Cập nhật lần cuối:**` với ngày hiện tại.
- Thêm/sửa đúng mục liên quan — không viết lại toàn bộ file.
- Giữ ngắn gọn: mỗi mục tối đa 1-2 dòng mô tả.
