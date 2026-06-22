import json
import numpy as np

files = {
    'v25 (ResNet1D)': r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v25_kq\kfold_V2\final_metrics.json',
    'v30 (CNN thuần)': r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30\kfold_v3\final_metrics.json',
    'v30_tcn (TCN)': r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn\kfoldv4_result\final_metrics.json'
}

labels = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']

def process_model(name, path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    md = f"\n## Chi tiết kết quả 17 lần huấn luyện: {name}\n\n"
    
    for scenario in ['S1_Elderly', 'S2_Young', 'S3_TrainSE_TestSA', 'S4_TrainSA_TestSE', 'S5_Both']:
        if scenario not in data: continue
        
        md += f"### Kịch bản: {scenario}\n\n"
        md += "| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |\n"
        md += "|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n"
        
        folds = data[scenario]
        if not isinstance(folds, list):
            folds = [folds] # convert single dict to list of 1 element
            
        for idx, f_data in enumerate(folds):
            run_name = f"Fold {idx+1}" if len(folds) > 1 else "1 Lần duy nhất"
            rep = f_data['report']
            
            # Precision row
            row_p = f"| **{run_name}** | **Precision (Tự tin)** "
            for L in labels:
                val = rep.get(L, {}).get('precision', 0.0)
                row_p += f"| {val:.4f} "
            row_p += "|\n"
            
            # Recall row
            row_r = f"| | **Recall (Độ phủ)** "
            for L in labels:
                val = rep.get(L, {}).get('recall', 0.0)
                row_r += f"| {val:.4f} "
            row_r += "|\n"
            
            # F1-score row
            row_f = f"| | **F1-Score** "
            for L in labels:
                val = rep.get(L, {}).get('f1-score', 0.0)
                row_f += f"| {val:.4f} "
            row_f += "|\n"
            
            md += row_p + row_r + row_f
        md += "\n"
    return md

all_md = ""
for name, path in files.items():
    all_md += process_model(name, path)

with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\detailed_tables.md', 'w', encoding='utf-8') as f:
    f.write(all_md)

print("Xong!")
