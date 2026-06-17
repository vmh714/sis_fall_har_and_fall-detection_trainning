---
trigger: always_on
description: "Quy định về việc sử dụng AI Models, Rate Limits và quản lý API Keys để tránh lỗi quá tải 429."
---

# Quy Tắc Sử Dụng AI Models và API Keys

## 1. Cấu hình API Keys
Khi cần tương tác với Google Gemini API (ví dụ qua script Python hoặc tool), Antigravity và Claude phải ưu tiên sử dụng danh sách các API key đã được cấp phép. 
Các API Key này được lưu tại: `datn_agent_skills/tools/api_key.txt`.

**Chiến lược xoay vòng (Key Rotation):**
Nếu thực thi script gặp lỗi `429 Too Many Requests` (hoặc Quota Exceeded), **bắt buộc** phải tự động xoay vòng sang sử dụng Key tiếp theo trong danh sách trên.

## 2. Định tuyến Mô hình (Model Routing) dựa trên Rate Limits
Tham khảo bảng Rate Limit tại `datn_agent_skills/tools/api_rate_limit.md` để chọn mô hình phù hợp nhằm tối ưu hóa quota:

- **Ưu tiên cao nhất cho tác vụ nền/lặp lại:** Sử dụng `Gemini 3.1 Flash Lite`. Đây là mô hình có Quota dồi dào nhất (15 Requests/Min, 500 Requests/Day).
- **Ưu tiên nhì:** Sử dụng `Gemini 2.5 Flash Lite` (10 Requests/Min, 20 Requests/Day).
- **Tác vụ thông thường/ít lặp lại:** Có thể sử dụng `Gemini 2.5 Flash`, `Gemini 3.5 Flash`, hoặc `Gemini 3 Flash` (Đều có 5 Requests/Min, 20 Requests/Day).
- **CẤM SỬ DỤNG:** Tuyệt đối không dùng các mô hình có quota = 0 (như `Gemini 3.1 Pro`, `Gemini 2.5 Pro`, `Gemini 2 Flash`, `Gemini 2 Flash Lite`).

*Việc tuân thủ quy tắc này giúp các agent (Antigravity/Claude) không bị sập pipeline do cạn kiệt API token.*