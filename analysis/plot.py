import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from datetime import datetime

log_folder = "/mnt/user/taosiyuan/projects/Giws/trainer_output/transformer/"

re_pattern = r"loss\s*=\s*([\d.]+)"
target_name = 'loss'.capitalize()
# re_pattern = r"lr\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
# target_name = 'lr'.capitalize()
re_pattern = r"bleu score:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
target_name = 'bleu score'.capitalize()


plt.switch_backend('agg')
plt.style.use('seaborn-v0_8')

mpl.font_manager.fontManager.addfont(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Helvetica-Regular.ttf'))

plt.rcParams['font.family'] = 'Helvetica Neue LT'
plt.rcParams['legend.fontsize'] = 12.0
plt.rcParams['axes.labelsize'] = 14.0
plt.rcParams['xtick.labelsize'] = 14.0
plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 200.0


def find_latest_date_folder(parent_dir):
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"parent folder doesn't exist: {parent_dir}")
    
    subdirs = [d for d in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, d))]
    
    date_folders = []
    for d in subdirs:
        match = re.match(r'^(\d{8})_', d)
        if match:
            try:
                date_str = match.group(1)
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                date_folders.append((date_obj, d))
            except ValueError:
                continue
    
    if not date_folders:
        return None
    
    date_folders_sorted = sorted(date_folders, key=lambda x: x[0], reverse=True)
    latest_folder = date_folders_sorted[0][1]
    
    return os.path.join(parent_dir, latest_folder)

def extract_logs(root_path):
    result = []
    log_dir = os.path.join(root_path, 'logs')
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_path = os.path.join(root, file)
                result.append(log_path)
    return result


log_folder = find_latest_date_folder(log_folder)
print(f"find latest log folder: {log_folder}")
logs = extract_logs(log_folder)
text = dict()
target = dict()

for name, path in enumerate(logs):
    with open(path, 'r') as f:
        text[f'rank{name}'] = f.read()

for name in text.keys():
    target[name] = re.findall(re_pattern, text[name])
    target[name] = [float(value) for value in target[name]]
del text

colors = ['blue', 'red', 'darkorange', 'purple', 'black', 'cyan', 'lime', 'gold']
for ids, name in enumerate(target.keys()):
    plt.plot([i for i in range(len(target[name]))], target[name], color=colors[ids], label=name)

plt.legend(frameon=False, loc='upper right')
plt.grid(linestyle='--')

plt.ylabel(target_name)
plt.legend()
plt.title(f'{target_name} Over Time')
plt.savefig('analysis/result.pdf',
            format='pdf',
            bbox_inches='tight',
            pad_inches=0.01)
