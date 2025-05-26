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
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="/mnt/user/taosiyuan/projects/Giws/trainer_output/")
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--target_name", type=str, default="loss")
    parser.add_argument('--offset', type=int, default=0)
    return parser.parse_args()


def find_latest_date_folder_by_offset(parent_dir, offset):
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
    target_name = str(args.target_name).capitalize()
    root_folder = os.path.join(args.log_folder, args.model)
    re_pattern = rf'{args.target_name}\s*:\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)%?'
    print(f"re_pattern: {re_pattern}")

    log_folder = find_latest_date_folder_by_offset(root_folder, args.offset)
    print(f"find latest log folder: {log_folder}")
    logs = extract_logs(log_folder)
    text = dict()
    target = dict()

    for name, path in enumerate(logs):
        with open(path, 'r') as f:
            text[f'rank{name}'] = f.read()

    for name in text.keys():
        import pdb;pdb.set_trace()
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


if __name__ == "__main__":
    args = parse()
    main(args)
