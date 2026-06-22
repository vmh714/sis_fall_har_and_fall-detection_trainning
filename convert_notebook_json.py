import json

with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\v30_opt_v2.py', 'r', encoding='utf-8') as f:
    code = f.read()

blocks = code.split('\n\n')

cells = []
for block in blocks:
    if block.strip():
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in block.split('\n')]
        })
        # Remove trailing newline from last line
        cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip('\n')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn_optimize_v2\train_v30_tcn_optimize_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook generated successfully as JSON!")
