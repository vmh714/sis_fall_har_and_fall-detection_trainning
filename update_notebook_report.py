import json

with open('SisFall_Colab_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the OutputReporter cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'class OutputReporter:' in ''.join(cell['source']):
        source_lines = cell['source']
        
        # We need to replace the evaluate_and_report method with the full one
        # Let's rebuild the class
        
        # Find where evaluate_and_report starts
        start_idx = -1
        for i, line in enumerate(source_lines):
            if 'def evaluate_and_report' in line:
                start_idx = i
                break
                
        if start_idx != -1:
            new_evaluate = [
                '    def evaluate_and_report(self, model, X_test, y_test, version="v_colab", fall_threshold=0.25):\n',
                '        if len(X_test) == 0: return\n',
                '        fall_idx = self.class_names.index(\'Fall\')\n',
                '        y_pred_probs = model.predict(X_test, batch_size=256)\n',
                '        y_pred = np.argmax(y_pred_probs, axis=1)\n',
                '        y_pred[y_pred_probs[:, fall_idx] >= fall_threshold] = fall_idx\n',
                '        \n',
                '        report_str = "\\n" + "="*50 + "\\n"\n',
                '        report_str += f"BÁO CÁO PHÂN LOẠI TẬP KIỂM THỬ - {version} (Threshold {fall_threshold})\\n"\n',
                '        report_str += "="*50 + "\\n"\n',
                '        \n',
                '        cls_report = classification_report(y_test, y_pred, target_names=self.class_names, digits=4)\n',
                '        report_str += cls_report + "\\n"\n',
                '        print(report_str)\n',
                '        \n',
                '        cm = confusion_matrix(y_test, y_pred)\n',
                '        plt.figure(figsize=(8,6))\n',
                '        plt.imshow(cm, cmap=plt.cm.Blues); plt.colorbar()\n',
                '        plt.xticks(np.arange(len(self.class_names)), self.class_names, rotation=45)\n',
                '        plt.yticks(np.arange(len(self.class_names)), self.class_names)\n',
                '        for i in range(cm.shape[0]):\n',
                '            for j in range(cm.shape[1]):\n',
                '                plt.text(j, i, format(cm[i,j],\'d\'), ha="center", color="white" if cm[i,j] > cm.max()/2. else "black")\n',
                '        plt.ylabel(\'Nhãn Thực Tế\')\n',
                '        plt.xlabel(\'Nhãn Dự Đoán\')\n',
                '        plt.tight_layout()\n',
                '        plt.savefig(self.out_dir / f\'confusion_matrix_{version}.png\')\n',
                '        plt.show()\n',
                '        \n',
                '        true_falls = np.sum(y_test == fall_idx)\n',
                '        detected_falls = cm[fall_idx, fall_idx]\n',
                '        recall_fall = (detected_falls / true_falls) * 100 if true_falls > 0 else 0\n',
                '        \n',
                '        eval_str = "\\n" + "="*50 + "\\n"\n',
                '        eval_str += "KẾT QUẢ ĐÁNH GIÁ CHUYÊN BIỆT LỚP TÉ NGÃ (FALL):\\n"\n',
                '        eval_str += f"  - Số ca ngã thực tế: {true_falls}\\n"\n',
                '        eval_str += f"  - Số ca ngã phát hiện đúng: {detected_falls}\\n"\n',
                '        eval_str += f"  - TỶ LỆ RECALL TÉ NGÃ: {recall_fall:.2f}%\\n"\n',
                '        eval_str += "="*50 + "\\n"\n',
                '        \n',
                '        print(eval_str)\n',
                '        report_str += eval_str\n',
                '        \n',
                '        with open(self.out_dir / f\'report_{version}.txt\', \'w\', encoding=\'utf-8\') as rf:\n',
                '            rf.write(report_str)\n',
                '        print(f"[*] Đã lưu báo cáo tại {self.out_dir / f\'report_{version}.txt\'}")\n'
            ]
            cell['source'] = source_lines[:start_idx] + new_evaluate

with open('SisFall_Colab_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Updated notebook successfully!")
