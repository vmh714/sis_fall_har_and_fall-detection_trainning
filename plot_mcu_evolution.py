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

# Data from Table 5.3 (MCU Performance)
# X: Inference Time (ms)
# Y: Accuracy (%)
# Size: Model Size (KB) * multiplier for visual
nodes = {
    'v30_tcn': {'time': 2014.30, 'acc': 95.80, 'size': 105.0, 'label': 'v30_tcn\n(Giãn nở)', 'group': 'tcn', 'offset': (0, -25)},
    'v30_tcn_opt': {'time': 137.57, 'acc': 95.60, 'size': 65.6, 'label': 'v30_tcn_opt\n(Bỏ giãn nở)', 'group': 'tcn', 'offset': (0, 15)},
    
    'v25': {'time': 22.27, 'acc': 90.60, 'size': 80.0, 'label': 'v25\n(ResNet cũ)', 'group': 'resnet', 'offset': (0, -25)},
    'v30_resnet': {'time': 20.16, 'acc': 95.00, 'size': 83.3, 'label': 'v30_resnet1d', 'group': 'resnet', 'offset': (0, 15)},
    
    'v30_lstm32': {'time': 94.46, 'acc': 94.50, 'size': 39.2, 'label': 'v30_lstm32\n(CNN+LSTM)', 'group': 'cnn_lstm', 'offset': (0, -25)},
    
    'v30': {'time': 5.14, 'acc': 93.00, 'size': 25.0, 'label': 'v30 (6 trục)', 'group': 'cnn', 'offset': (-30, -20)},
    'v31': {'time': 5.13, 'acc': 92.60, 'size': 24.9, 'label': 'v31 (4 trục)', 'group': 'cnn', 'offset': (0, -25)},
    'v32_w128': {'time': 3.72, 'acc': 92.20, 'size': 24.5, 'label': 'w128', 'group': 'cnn', 'offset': (15, 0)},
    'v32_w256': {'time': 6.04, 'acc': 92.80, 'size': 24.5, 'label': 'w256', 'group': 'cnn', 'offset': (15, 0)},
    
    'v30_opt': {'time': 11.20, 'acc': 95.40, 'size': 55.8, 'label': 'v30_optimize\n(Đề xuất cuối)', 'group': 'opt', 'offset': (0, 20)}
}

# Group styling
groups = {
    'tcn': {'color': '#e74c3c', 'marker': 'o', 'label': 'Nhánh TCN'},
    'resnet': {'color': '#3498db', 'marker': 's', 'label': 'Nhánh ResNet'},
    'cnn_lstm': {'color': '#9b59b6', 'marker': 'D', 'label': 'Nhánh lai CNN-LSTM'},
    'cnn': {'color': '#2ecc71', 'marker': '^', 'label': 'Nhánh CNN thuần (Siêu nhanh)'},
    'opt': {'color': '#f1c40f', 'marker': '*', 'label': 'Mô hình Đề xuất'}
}

edges = [
    # TCN evolution
    ('v30_tcn', 'v30_tcn_opt', 'solid', '#e74c3c', 'Thay bằng Conv thường'),
    
    # ResNet evolution
    ('v25', 'v30_resnet', 'solid', '#3498db', 'Gen 2'),
    ('v25', 'v30', 'solid', '#3498db', 'Bỏ SE Block\nChuyển sang CNN'),
    
    # CNN branch out
    ('v30', 'v31', 'solid', '#2ecc71', 'Rút 4 trục'),
    ('v30', 'v32_w128', 'solid', '#2ecc71', ''),
    ('v30', 'v32_w256', 'solid', '#2ecc71', ''),
    
    # Opt branch
    ('v30', 'v30_opt', 'solid', '#f39c12', 'Cải tiến Head (Giữ trục tgian)')
]

fig, ax = plt.subplots(figsize=(13, 8))

# Draw edges
for u, v, style, color, text in edges:
    x1, y1 = nodes[u]['time'], nodes[u]['acc']
    x2, y2 = nodes[v]['time'], nodes[v]['acc']
    
    rad = 0.1
    
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle=f"arc3,rad={rad}",
                                    color=color,
                                    linestyle=style,
                                    linewidth=2,
                                    arrowstyle="->",
                                    mutation_scale=15,
                                    alpha=0.6)
    ax.add_patch(arrow)
    
    # Add text to arrow (calculate linear midpoint on log scale if needed)
    if text:
        cx = np.sqrt(x1 * x2) # geometric mean for log scale midpoint
        cy = (y1 + y2) / 2 + 0.15
        
        ax.text(cx, cy, text, color=color, fontsize=9, fontweight='bold',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# Draw nodes
plotted_groups = set()
for key, node in nodes.items():
    grp = groups[node['group']]
    
    # Scale size: base size 150 + node size * some multiplier
    size = 100 + node['size'] * 5 if node['group'] != 'opt' else 800
    edgecolor = '#d35400' if node['group'] == 'opt' else 'white'
    
    lbl = grp['label'] if node['group'] not in plotted_groups else ""
    plotted_groups.add(node['group'])
    
    ax.scatter(node['time'], node['acc'], 
               color=grp['color'], marker=grp['marker'], 
               s=size, edgecolor=edgecolor, linewidth=1.5, zorder=5, label=lbl)
    
    # Label the node
    ax.annotate(node['label'], 
                xy=(node['time'], node['acc']), 
                xytext=node['offset'], 
                textcoords='offset points', 
                ha='center', va='center', 
                fontsize=9, fontweight='bold', zorder=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.3))

# Formatting
ax.set_title('Sơ đồ Đánh đổi Hiệu năng \& Tiến hóa trên Vi điều khiển ESP32-S3', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Thời gian suy luận - Inference Time (ms) [Log Scale] $\\rightarrow$ Chậm hơn', fontsize=13, fontweight='bold')
ax.set_ylabel('Độ chính xác - Accuracy (%)', fontsize=13, fontweight='bold')

# Use Log Scale for X axis because of the huge gap (3ms to 2000ms)
ax.set_xscale('log')
ax.set_xticks([3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylim(89.5, 96.5)

# Highlight realtime constraint (< 200ms)
ax.axvspan(1, 200, color='#2ecc71', alpha=0.1, zorder=0, label='Vùng thời gian thực khả thi (< 200ms)')
ax.axvspan(200, 3000, color='#e74c3c', alpha=0.05, zorder=0, label='Quá chậm')

# Legend
ax.legend(loc='lower left', framealpha=0.9, fontsize=11)

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Save the figure
output_path = r'd:\datn\report\Do_an_tot_nghiep_Vu_Manh_Hung\Hinhve\mcu_evolution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved MCU evolution chart to {output_path}")
