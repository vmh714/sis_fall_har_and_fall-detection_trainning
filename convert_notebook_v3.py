import nbformat
from nbformat.v4 import new_notebook, new_code_cell

with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\v30_opt_v2.py', 'r', encoding='utf-8') as f:
    code = f.read()

blocks = code.split('\n\n')
cells = [new_code_cell(block) for block in blocks if block.strip()]

nb = new_notebook(cells=cells)
with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn_optimize_v2\train_v30_tcn_optimize_v2.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully!")
