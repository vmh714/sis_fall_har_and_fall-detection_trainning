import json
import numpy as np

files = {
    'v30 (CNN)': r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30\kfold_v3\final_metrics.json',
    'v30_tcn (TCN)': r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn\kfoldv4_result\final_metrics.json',
    'v25 (ResNet1D)': r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v25_kq\kfold_V2\final_metrics.json'
}

def extract_metrics(data):
    results = {}
    for scenario, folds in data.items():
        if isinstance(folds, list):
            accs = [f['acc'] for f in folds]
            f1s = [f['report']['macro avg']['f1-score'] for f in folds]
            fall_recalls = [f['report']['Fall']['recall'] for f in folds]
            trans_f1s = [f['report']['Trans']['f1-score'] for f in folds]
            results[scenario] = {
                'acc': np.mean(accs),
                'f1': np.mean(f1s),
                'fall_recall': np.mean(fall_recalls),
                'trans_f1': np.mean(trans_f1s)
            }
        else:
            f = folds
            results[scenario] = {
                'acc': f['acc'],
                'f1': f['report']['macro avg']['f1-score'],
                'fall_recall': f['report']['Fall']['recall'],
                'trans_f1': f['report']['Trans']['f1-score']
            }
    return results

for name, path in files.items():
    with open(path, 'r') as f:
        data = json.load(f)
    metrics = extract_metrics(data)
    print(f"\n=== {name} ===")
    for sc, m in metrics.items():
        print(f"{sc:20}: Acc={m['acc']:.4f}, Macro F1={m['f1']:.4f}, Fall Recall={m['fall_recall']:.4f}, Trans F1={m['trans_f1']:.4f}")
