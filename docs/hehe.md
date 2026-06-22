Đủ thông tin rồi. Đây là toàn bộ dữ liệu thực tôi đã thu thập, sẵn sàng dùng cho báo cáo:

Dữ liệu thực từ project — tổng hợp cho báo cáo
Bảng 1: So sánh các phiên bản mô hình (tab:model_comparison)
Phiên bản	Kiến trúc	Nhãn	Acc (%)	F1-Fall	F1-Trans	Macro-F1
v1	CNN-LSTM baseline	4	91.57	0.9755	—	0.9294
v8	TCN dilated causal	4	91.34	0.9667	—	0.9303
v22	TCN + SE + Dual Pooling	5	91.56	0.9786	0.8554	0.9267
v25	1D ResNet + SeparableConv	5	91.01	0.9784	0.8365	0.9238
Nguồn file thực: train_v1_ketqua/report.txt, train_v8_kq/report_v8.txt, train_v22_kq/report_v22.txt, train_v25_kq/report_v25.txt

Lưu ý quan trọng: v1 và v8 dùng 4 nhãn (Walk, Run, Static/ADL, Fall) — không có F1-Trans. v22 trở đi mới có 5 nhãn.

Bảng 2: Hiệu năng MCU (tab:mcu_perf)
Từ report_v25_firmware.txt:

Chỉ số	Giá trị
Kích thước model TFLite (INT8)	~205 kB (210,192 bytes thực tế)
Tensor arena cấu hình	100 kB (PSRAM)
Tensor arena thực dùng	28.2 kB (28,844 bytes)
Thời gian inference trung bình	70.23 ms
Thời gian min / max	70.07 / 70.32 ms
Kết quả firmware (1,200 mẫu)	Acc 90.33%, Fall Recall 99.00%
Bảng 3: Kiến trúc layer-by-layer v25 (cho Section 5.2)
Block	Layer	Output Shape	Params
Stem	Conv1D(16, k=3, s=2) + BN + ReLU6	(100, 16)	—
Block 1	SepConv(16, k=3, s=1) × 2 + SE + Residual	(100, 16)	—
Block 2	SepConv(32, k=3, s=2) × 2 + SE + Residual	(50, 32)	—
Block 3	SepConv(32, k=3, s=1) × 2 + SE + Residual	(50, 32)	—
Block 4	SepConv(64, k=3, s=2) × 2 + SE + Residual	(25, 64)	—
Head	GAP + GMP → Concat → Dropout(0.3) → Dense(5)	(5,)	—
Tổng			19,253 (18,453 trainable)
File: train_v25_kq/report_v25.txt dòng đầu chứa model.summary() output đầy đủ

Bảng 4: Hyperparameter training v25 (cho tab:training_config)
Tham số	v25	v26 (tham chiếu)
Optimizer	Adam	Adam
Learning rate	0.001	0.001
Batch size	64	256
Max epochs	100	100
EarlyStopping patience	15	15
ReduceLROnPlateau factor/patience	0.5 / 5	0.5 / 5
Loss function	CategoricalCrossentropy(label_smoothing=0.1)	CategoricalFocalCrossentropy(gamma=2.0)
Fall class weight	×3.0	balanced (sklearn auto)
Fall decision threshold	0.25	default (0.5)
Ánh xạ nhãn ↔ mã SisFall (cho Section 5.1 tab:label_mapping)
Từ ml_pipeline.py:parse_filename_info():

Nhãn	Mã hoạt động SisFall	Ghi chú
Walk	D01, D02, D05, D06	Đi bộ chậm/nhanh, leo/xuống cầu thang
Run	D03, D04	Chạy bộ chậm/nhanh
Idle	_StandSit_, _Lie_	Xác định qua peak → đoạn tĩnh
Trans	_Trans_	Xác định qua peak → vùng chuyển tiếp
Fall	F01–F15	Toàn bộ sự kiện ngã
Pipeline 5 nhãn: Với D07–D16 (ngồi, nằm, cúi,...), dùng peak detection trên SVM → cắt 3 window quanh đỉnh gán nhãn Trans, đoạn tĩnh xa đỉnh xác định theo hướng vector trọng lực → Idle_StandSit hay Idle_Lie (sau merge thành Idle).

Subject Split LSO (từ ml_pipeline.py:18-20)
Tập	Subjects	Số lượng
Train	SA01–SA18 + SE01–SE08	26
Val	SA19–SA21 + SE09–SE11	6
Test	SA22–SA23 + SE12–SE15	6
Section 5.3 — PCNT status
Kết luận: chưa implement PCNT. File sis_fall_firmware_inference.c hiện là firmware đánh giá qua UART (PC gửi window → MCU chạy inference trả kết quả). Sampling thực tế dùng vTaskDelay(pdMS_TO_TICKS(10)) cho UART polling, không phải hardware timer ISR hay PCNT. → Nên điều chỉnh thì động từ từ "đã triển khai PCNT" → "được thiết kế để dùng hardware timer" trong văn bản.

Giờ bạn muốn tôi bắt đầu điền thẳng vào các file .tex (Chương 4.3 và Chương 5) không? Cần biết path đến folder REPORT của bạn — nếu nó ở workspace khác thì bạn cần add folder đó vào.


Full table — tất cả các version
Ver	Classes	Acc%	F1-Fall	F1-Trans	Macro-F1	Ghi chú
v1	4	91.57	0.9755	—	0.9294	CNN-LSTM baseline
v2	4	91.49	0.9591	—	0.9261	↔ v1 gần như ngang
v3	4	90.00	0.9614	—	0.9104	thụt lùi
v4	4	92.41	0.9774	—	0.9353	peak 4-class
v5	4	91.02	0.9549	—	0.9213	
v6	4	92.34	0.9766	—	0.9381	≈ v4
v7	4	90.39	0.9721	—	0.9229	
v8	4	91.34	0.9667	—	0.9303	TCN dilated causal
v8_legacy	4	91.49	0.9275	—	0.9200	variant
v9	4	91.49	0.9635	—	0.9255	
v10	4	92.46	0.9658	—	0.9367	
v11	4	91.40	0.9281	—	0.9194	dataset mới (8116)
v11_cu	4	86.96	0.9504	—	0.8980	thất bại
v12	4	89.87	0.9454	—	0.9159	
v12_6ch	4	91.86	0.9391	—	0.9273	6ch variant
v12_6ch_opt	4	89.79	0.8893	—	0.9009	
v12_7ch	4	91.30	0.9507	—	0.9243	7ch variant
v13	4	85.49	0.7452	—	0.8468	thất bại nặng
v14	4	92.16	0.9626	—	0.9335	
v14_6ch	4	89.44	0.8801	—	0.9000	
v14_7ch	4	90.20	0.8675	—	0.9038	
v15	4	91.54	0.9392	—	0.9235	
v16	6	87.89	0.9492	0.7210	0.8768	thử 6 nhãn
v17	6	88.20	0.9831	0.7215	0.8812	thử 6 nhãn
v18	5	89.89	0.9759	0.7820	0.9092	first 5-class
v19	5	82.38	0.9231	0.7746	0.8527	thất bại
v20	5	89.36	0.9742	0.7960	0.9056	
v21	5	90.68	0.9728	0.8193	0.9178	
v22	5	91.56	0.9786	0.8554	0.9267	TCN+SE+DualPool
v23	5	91.33	0.9832	0.8444	0.9273	best Fall F1
v23_firmware	5	54.17	0.6620	0.4312	0.4025	bug INT8 chưa fix
v24	5	90.75	0.9838	0.8442	0.9233	
v24_firmware	5	90.33	0.9950	0.8374	0.9132	sau khi fix INT8
v25	5	91.01	0.9784	0.8365	0.9238	ResNet-1D final
v25_firmware	5	90.33	0.9950	0.8374	0.9132	
v25_v2	5	90.16	0.9758	0.8289	0.9166	minor tuning
v26	5	87.95	0.9454	0.7955	0.8920	Focal Loss
v26_v2	5	88.02	0.9587	0.7809	0.8923	
v26_v3	5	88.54	0.9514	0.8041	0.8964	best v26 variant
Gợi ý lọc — các version nên bỏ qua
Nhóm 1 — Biến thể kênh (loại bỏ hoàn toàn):
v8_legacy, v12_6ch, v12_6ch_opt, v12_7ch, v14_6ch, v14_7ch → thử nghiệm số kênh, không phải milestone kiến trúc.

Nhóm 2 — Thất bại không đáng nhắc trong bảng (có thể nhắc trong văn bản):
v11_cu (86.96%), v13 (85.49% — MaxPooling fail), v19 (82.38%) → dùng để giải thích "thử sai" trong text, không cần hàng riêng.

Nhóm 3 — Mờ nhạt / gần như trùng nhau:
v2, v3, v5, v7, v9, v10 (quanh quẩn 90–92%, không có đột phá kiến trúc), v25_v2, v26_v2 → bỏ, giữ best của mỗi cụm.

Nhóm 4 — Firmware results → bảng riêng:
v23_firmware, v24_firmware, v25_firmware → không nên trộn vào bảng PC-side, để sang tab:mcu_perf.

Nhóm 5 — 6-class thực nghiệm:
v16, v17 → nên nhắc 1 câu trong text ("thử 6 nhãn, Macro-F1 0.88 thấp hơn 5-class"), không cần hàng bảng.

Đề xuất bảng cuối (~8 hàng, kể chuyện rõ ràng)
Ver	Mô tả kiến trúc	#Nhãn	Acc%	F1-Fall	F1-Trans	Macro-F1
v1	CNN-LSTM baseline	4	91.57	0.9755	—	0.9294
v4	TCN (peak 4-class)	4	92.41	0.9774	—	0.9353
v8	TCN dilated causal	4	91.34	0.9667	—	0.9303
v18	TCN — chuyển sang 5 nhãn	5	89.89	0.9759	0.7820	0.9092
v22	TCN + SE + Dual Pooling	5	91.56	0.9786	0.8554	0.9267
v23	TCN + ngưỡng Fall tối ưu	5	91.33	0.9832	0.8444	0.9273
v25	ResNet-1D + SeparableConv	5	91.01	0.9784	0.8365	0.9238
v26_v3	ResNet-1D + Focal Loss	5	88.54	0.9514	0.8041	0.8964
Câu chuyện rõ: 4-class → peak → chuyển 5 nhãn (giảm nhẹ) → cải thiện dần → đạt đỉnh Fall ở v23 → v25 tối ưu MCU → v26 thử Focal Loss.