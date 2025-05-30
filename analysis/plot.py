import re
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


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



def parse():
    def parse_flexible_list(value):
        if ',' in value:
            return [int(x) for x in value.split(',')]
        else:
            return [int(value)]  # 单个值转为单元素列表
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="/mnt/user/taosiyuan/projects/Giws/trainer_output/")
    parser.add_argument("--model", type=str, default="transformer", help="e.g., vit/lstm, etc.")
    parser.add_argument("--indicator", type=str, default="loss", help="The name of your target indicator you want to plot.")
    parser.add_argument("--sep", type=str, default="=", help="Sep char between the indicator and value. e.g., :, =.")
    parser.add_argument('--ranks', type=parse_flexible_list, default=None, help='Which rank\'s output you want to plot. List of numbers (e.g., "0" or "0,1")')
    parser.add_argument("--offset", type=int, default=0, help="Offset of the folder in the date list, default 0.(e.g., -1, 4)")
    return parser.parse_args()


def find_latest_date_folder_by_offset(parent_dir, offset):
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"parent folder doesn't exist: {parent_dir}")
    
    subdirs = [d for d in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, d))]
    
    date_folders = []

    for d in subdirs:
        match = re.match(r'^(\d{8}_\d{6})', d)
        if match:
            try:
                date_str = match.group(1)
                date_obj = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                date_folders.append((date_obj, d))
            except ValueError:
                continue
    
    if not date_folders:
        return None
    
    
    date_folders_sorted = sorted(date_folders, key=lambda x: x[0], reverse=True)
    latest_folder_by_offset = date_folders_sorted[offset][1]
    
    return os.path.join(parent_dir, latest_folder_by_offset)

def extract_logs(root_path):
    result = []
    log_dir = os.path.join(root_path, 'logs')
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_path = os.path.join(root, file)
                result.append(log_path)
    return result


def main(args):
    indicator = str(args.indicator).capitalize()
    root_folder = os.path.join(args.log_folder, args.model)
    re_pattern = rf'\b{args.indicator}\s*{args.sep}\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)%?'
    ranks = args.ranks
    print(f"re_pattern: {re_pattern}")
    
    log_folder = find_latest_date_folder_by_offset(root_folder, args.offset)
    print(f"find latest log folder: {log_folder}")
    logs = extract_logs(log_folder)
    text = dict()
    target = dict()

    for name, path in enumerate(logs):
        with open(path, 'r') as f:
            text[f'rank{name}'] = f.read()

    if ranks is None:
        ranks = list(text.keys())
    else:
        ranks = [f'rank{r}' for r in ranks]
    
    target = {
        name: [float(value) for value in re.findall(re_pattern, text[name])]
        for name in text 
        if re.findall(re_pattern, text[name]) and name in ranks
    }
    del text

    if len(target.keys()) == 0:
        raise ValueError(f"No data extracted! target[{name}] = {target[name]}. Please check your log file or the re pattern.")

    colors = ['blue', 'red', 'darkorange', 'purple', 'black', 'cyan', 'lime', 'gold']
    for ids, name in enumerate(target.keys()):
        plt.plot([i for i in range(len(target[name]))], target[name], color=colors[ids], label=name)

    plt.legend(frameon=False, loc='upper right')
    plt.grid(linestyle='--')

    plt.ylabel(indicator)
    plt.legend()
    plt.title(f'{indicator} Over Time')
    plt.savefig('analysis/result.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)


if __name__ == "__main__":
    args = parse()
    main(args)
