import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

plt.switch_backend('agg')
plt.style.use('seaborn-paper')

plt.rcParams['legend.fontsize'] = 12.0
plt.rcParams['axes.labelsize'] = 14.0
plt.rcParams['xtick.labelsize'] = 14.0
plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 200.0

def extract_logs(root_path):
    result = []
    folders = os.listdir(root_path)
    print(folders)
    for folder in folders:
        root = os.path.join(root_path, folder)
        log_dir = os.path.join(root, 'logs')
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith(".log"):
                    log_path = os.path.join(root, file)
                    result.append((folder, log_path))
    return result


logs = extract_logs('./scripts/output')
text = dict()
acc = dict()

for name, path in logs:
    with open(path, 'r') as f:
        text[name] = f.read()

for name in text.keys():
    acc[name] = re.findall(r"Average Accuracy: (\d+\.\d+)", text[name])
    acc[name] = [float(value) for value in acc[name]]
del text

colors = ['steelblue', 'lightcoral', 'gold', 'palegreen', 'skyblue']
for ids, name in enumerate(acc.keys()):
    plt.plot([i for i in range(len(acc[name]))], acc[name], color=colors[ids], label=name)


plt.xlabel('Evaluation Step')
plt.ylabel('Average Accuracy')
plt.legend()
plt.title('Average Accuracy Over Time')
plt.savefig('analysis/result.png')
