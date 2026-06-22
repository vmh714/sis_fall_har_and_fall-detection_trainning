import glob, os, numpy as np
import pandas as pd

DIR = r"d:\New folder\sis_fall_har_and_fall-detection_trainning\SisFall_dataset_Windowed_v30_TEST"
files = glob.glob(os.path.join(DIR, "*.csv"))

classes = {
    'Walk': [],
    'Run': [],
    'Idle_Lie': [],
    'Idle_StandSit': [],
    'Trans': [],
    'Fall': []
}

for f in files:
    name = os.path.basename(f)
    if '_Walk_' in name:
        c = 'Walk'
    elif '_Run_' in name:
        c = 'Run'
    elif '_Idle_Lie_' in name:
        c = 'Idle_Lie'
    elif '_Idle_StandSit_' in name:
        c = 'Idle_StandSit'
    elif '_Trans_' in name:
        c = 'Trans'
    elif '_Fall_' in name:
        c = 'Fall'
    else:
        continue
    classes[c].append(f)

with open("window_percentiles_full.md", "w", encoding="utf-8") as out:
    out.write("#### 1. Bảng Phân Tích Bách Phân Vị Gia Tốc (Accel) Theo Từng Hành Động\n")
    out.write("| Nhãn (Class) | X-P50 | X-P95 | X-P99.9 | Y-P50 | Y-P95 | Y-P99.9 | Z-P50 | Z-P95 | Z-P99.9 | SVM-P50 | SVM-P95 | SVM-P99.9 |\n")
    out.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    results = {}
    for c, flist in classes.items():
        if not flist: continue
        np.random.seed(42)
        if len(flist) > 1000:
            flist = np.random.choice(flist, 1000, replace=False)
            
        ax_all, ay_all, az_all, svm_all = [], [], [], []
        gx_all, gy_all, gz_all, rms_all = [], [], [], []
        
        for f in flist:
            try:
                df = pd.read_csv(f)
                ax, ay, az = np.abs(df['ax'].values), np.abs(df['ay'].values), np.abs(df['az'].values)
                gx, gy, gz = np.abs(df['gx'].values), np.abs(df['gy'].values), np.abs(df['gz'].values)
                
                svm = np.sqrt(ax**2 + ay**2 + az**2)
                rms = np.sqrt(gx**2 + gy**2 + gz**2)
                
                ax_all.append(ax); ay_all.append(ay); az_all.append(az); svm_all.append(svm)
                gx_all.append(gx); gy_all.append(gy); gz_all.append(gz); rms_all.append(rms)
            except:
                pass
                
        if ax_all:
            results[c] = {
                'ax': np.concatenate(ax_all), 'ay': np.concatenate(ay_all), 'az': np.concatenate(az_all), 'svm': np.concatenate(svm_all),
                'gx': np.concatenate(gx_all), 'gy': np.concatenate(gy_all), 'gz': np.concatenate(gz_all), 'rms': np.concatenate(rms_all)
            }

    for c in ['Walk', 'Run', 'Idle_Lie', 'Idle_StandSit', 'Trans', 'Fall']:
        if c not in results: continue
        d = results[c]
        row = f"| **{c}** | "
        for k in ['ax', 'ay', 'az', 'svm']:
            p50 = np.percentile(d[k], 50)
            p95 = np.percentile(d[k], 95)
            p999 = np.percentile(d[k], 99.9)
            row += f"{p50:.2f} | {p95:.2f} | {p999:.2f} | "
        out.write(row.strip() + "\n")

    out.write("\n#### 2. Bảng Phân Tích Bách Phân Vị Vận Tốc Góc (Gyro) Theo Từng Hành Động\n")
    out.write("| Nhãn (Class) | X-P50 | X-P95 | X-P99.9 | Y-P50 | Y-P95 | Y-P99.9 | Z-P50 | Z-P95 | Z-P99.9 | RMS-P50 | RMS-P95 | RMS-P99.9 |\n")
    out.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    for c in ['Walk', 'Run', 'Idle_Lie', 'Idle_StandSit', 'Trans', 'Fall']:
        if c not in results: continue
        d = results[c]
        row = f"| **{c}** | "
        for k in ['gx', 'gy', 'gz', 'rms']:
            p50 = np.percentile(d[k], 50)
            p95 = np.percentile(d[k], 95)
            p999 = np.percentile(d[k], 99.9)
            row += f"{p50:.2f} | {p95:.2f} | {p999:.2f} | "
        out.write(row.strip() + "\n")

print("Done.")
