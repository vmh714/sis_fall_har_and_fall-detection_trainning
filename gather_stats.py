import glob, os, re

models = ['v30_lstm32', 'v30', 'v30_resnet1d', 'v30_tcn', 'v30_tcn_optimize', 'v30_tcn_optimize_v2']
base_dir = r'd:\New folder\sis_fall_har_and_fall-detection_trainning'

for m in models:
    print(f"\n[{m.upper()}]")
    
    # Find tflite
    tflite_files = glob.glob(os.path.join(base_dir, f'**/{m}**/*.tflite'), recursive=True) + \
                   glob.glob(os.path.join(base_dir, f'**/output_{m}**/*.tflite'), recursive=True)
    for tf_file in set(tflite_files):
        print(f"  TFLite: {os.path.basename(tf_file)} -> {os.path.getsize(tf_file)} bytes")
        
    # Find firmware report
    fw_reports = glob.glob(os.path.join(base_dir, f'firmware_test_tool/report_{m}*.txt'))
    for fwr in fw_reports:
        with open(fwr, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract time and RAM
            time_match = re.search(r'Thời gian Inference trung bình: ([\d\.]+) ms', content)
            ram_match = re.search(r'Tensor Arena \(RAM\) sử dụng: (\d+) bytes', content)
            recall_match = re.search(r'TỶ LỆ RECALL TÉ NGÃ.*?: ([\d\.]+)%', content)
            f1_match = re.search(r'Trans\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)', content)
            acc_match = re.search(r'accuracy\s+([\d\.]+)', content)
            print(f"  Firmware Report: {os.path.basename(fwr)}")
            if time_match: print(f"    Time: {time_match.group(1)} ms")
            if ram_match: print(f"    RAM: {ram_match.group(1)} bytes")
            if recall_match: print(f"    Fall Recall: {recall_match.group(1)}%")
            if f1_match: print(f"    Trans F1: {f1_match.group(1)}")
            if acc_match: print(f"    Accuracy: {acc_match.group(1)}")
