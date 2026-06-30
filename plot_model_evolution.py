import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib

# Use a non-interactive backend
matplotlib.use('Agg')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Data definition
nodes = {
    # TCN Group
    'v18': {'acc': 89.89, 'f1': 0.9092, 'label': 'v18\n(TCN baseline)', 'group': 'tcn', 'offset': (0, -25)},
    'v22': {'acc': 91.56, 'f1': 0.9267, 'label': 'v22\n(TCN + SE)', 'group': 'tcn', 'offset': (-10, 15)},
    'v30_tcn': {'acc': 96.02, 'f1': 0.9596, 'label': 'v30_tcn\n(TCN tối ưu)', 'group': 'tcn', 'offset': (10, 15)},
    
    # ResNet Group
    'v25': {'acc': 91.01, 'f1': 0.9238, 'label': 'v25\n(ResNet-1D)', 'group': 'resnet', 'offset': (0, -25)},
    'v30_resnet': {'acc': 94.15, 'f1': 0.9367, 'label': 'v30_resnet1d', 'group': 'resnet', 'offset': (0, 15)},
    
    # CNN Group
    'v30': {'acc': 93.43, 'f1': 0.9259, 'label': 'v30\n(CNN baseline)', 'group': 'cnn', 'offset': (0, -25)},
    'v31': {'acc': 92.37, 'f1': 0.9154, 'label': 'v31\n(CNN 4-trục)', 'group': 'cnn', 'offset': (0, -25)},
    
    # CNN-LSTM Group
    'v1': {'acc': 91.57, 'f1': 0.9294, 'label': 'v1\n(CNN-LSTM)', 'group': 'cnn_lstm', 'offset': (-35, 10)},
    'v30_lstm32': {'acc': 92.74, 'f1': 0.9235, 'label': 'v30_lstm32\n(CNN-LSTM)', 'group': 'cnn_lstm', 'offset': (-35, 10)},
    
    # Proposed
    'v30_opt': {'acc': 95.33, 'f1': 0.9524, 'label': 'v30_optimize\n(Đề xuất cuối)', 'group': 'opt', 'offset': (-20, -30)}
}

# Group styling
groups = {
    'tcn': {'color': '#e74c3c', 'marker': 'o', 'label': 'Nhánh TCN (Giãn nở, Chậm)'},
    'resnet': {'color': '#3498db', 'marker': 's', 'label': 'Nhánh ResNet (Có SE block)'},
    'cnn': {'color': '#2ecc71', 'marker': '^', 'label': 'Nhánh CNN (Thuần ESP-NN, Nhanh)'},
    'cnn_lstm': {'color': '#9b59b6', 'marker': 'D', 'label': 'Nhánh CNN-LSTM (Khởi đầu)'},
    'opt': {'color': '#f1c40f', 'marker': '*', 'label': 'Mô hình Đề xuất', 'size': 600, 'edgecolor': '#d35400'}
}

edges = [
    # CNN-LSTM evolution
    ('v1', 'v18', 'solid', '#9b59b6', 'Chuyển sang TCN'),
    ('v1', 'v30_lstm32', 'solid', '#9b59b6', 'Nâng cấp lên 5 nhãn'),
    
    # TCN evolution
    ('v18', 'v22', 'solid', '#e74c3c', 'Tối ưu TCN'),
    ('v22', 'v30_tcn', 'solid', '#e74c3c', 'TCN Gen2'),
    
    # ResNet branch out
    ('v22', 'v25', 'solid', '#3498db', 'Bỏ giãn nở'),
    ('v25', 'v30_resnet', 'solid', '#3498db', 'ResNet Gen2'),
    
    # CNN branch out
    ('v25', 'v30', 'solid', '#2ecc71', 'Bỏ SE block'),
    ('v30', 'v31', 'solid', '#2ecc71', 'Rút gọn (4 trục)'),
    
    # Opt branch
    ('v30', 'v30_opt', 'solid', '#f39c12', 'Cải tiến Head'),
    
    # Inspiration/Knowledge distillation
    ('v30_tcn', 'v30_opt', 'dashed', '#7f8c8d', 'Kế thừa ý tưởng\ngiữ trục thời gian')
]

fig, ax = plt.subplots(figsize=(13, 8))

# Draw edges
for u, v, style, color, text in edges:
    x1, y1 = nodes[u]['acc'], nodes[u]['f1']
    x2, y2 = nodes[v]['acc'], nodes[v]['f1']
    
    # Adjust curvature based on edge to prevent overlap
    rad = 0.1
    if style == 'dashed':
        rad = -0.2
    if u == 'v1' and v == 'v30_lstm32':
        rad = 0.2
    
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle=f"arc3,rad={rad}",
                                    color=color,
                                    linestyle=style,
                                    linewidth=2,
                                    arrowstyle="->",
                                    mutation_scale=15,
                                    alpha=0.7)
    ax.add_patch(arrow)
    
    # Add text to arrow
    if text:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Adjust text position for specific arrows
        if style == 'dashed':
            cx += 0.2
            cy += 0.005
        elif u == 'v1' and v == 'v18':
            cx += 0.05
            cy -= 0.008
        elif u == 'v1' and v == 'v30_lstm32':
            cx -= 0.1
            cy += 0.005
        else:
            cx -= 0.1
            cy += 0.002
            
        ax.text(cx, cy, text, color=color, fontsize=9, fontweight='bold',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# Draw nodes
plotted_groups = set()
for key, node in nodes.items():
    grp = groups[node['group']]
    size = grp.get('size', 150)
    edgecolor = grp.get('edgecolor', 'white')
    
    # Only label the group once for the legend
    lbl = grp['label'] if node['group'] not in plotted_groups else ""
    plotted_groups.add(node['group'])
    
    ax.scatter(node['acc'], node['f1'], 
               color=grp['color'], marker=grp['marker'], 
               s=size, edgecolor=edgecolor, linewidth=1.5, zorder=5, label=lbl)
    
    # Label the node
    ax.annotate(node['label'], 
                xy=(node['acc'], node['f1']), 
                xytext=node['offset'], 
                textcoords='offset points', 
                ha='center', va='center', 
                fontsize=10, fontweight='bold', zorder=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

# Formatting
ax.set_title('Sơ đồ Phân nhánh Tiến hóa Kiến trúc Mô hình (Model Evolution)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Độ chính xác - Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Điểm Macro-F1', fontsize=13, fontweight='bold')

ax.set_xlim(88.5, 97)
ax.set_ylim(0.90, 0.97)

# Legend
ax.legend(loc='lower right', framealpha=0.9, fontsize=11, title="Các hướng tiếp cận", title_fontsize=12)

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Save the figure
output_path = r'd:\datn\report\Do_an_tot_nghiep_Vu_Manh_Hung\Hinhve\model_evolution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved evolution network chart to {output_path}")
