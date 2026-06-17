import sys
import glob
sys.stdout.reconfigure(encoding='utf-8')
import json
brain_path = r'C:\Users\vuman\.gemini\antigravity-ide\brain\*\.system_generated\logs\transcript.jsonl'
keywords = ['clean', 'tiền xử lý', 'preprocess', 'z-score', 'overlap', 'dataset']
for path in glob.glob(brain_path):
    print(f"--- Checking conversation: {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('type') == 'USER_INPUT':
                    content = data.get('content', '')
                    if any(k in content.lower() for k in keywords):
                        print(f"--- STEP {data.get('step_index')} ---")
                        print(content[:500])
                        print("-" * 50)
    except Exception as e:
        print("Error reading", path, e)

