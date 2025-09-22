import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
# fm.fontManager.clear_cached_fonts()
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # Replace with the actual path
font_prop = fm.FontProperties(fname=font_path)
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = font_prop.get_name()
# 如果字体文件路径没有问题，可以清理缓存
# fm.fontManager.clear_cached_fonts()

def datavisual(data_file, output_file):
    # 检查文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"The data file {data_file} does not exist.")

    # 初始化字典以存储数据
    data_dict = {
        'Smeasure': {'Task': [], 'Dataset': [], 'Value': []},
        'meanEm': {'Task': [], 'Dataset': [], 'Value': []},
        'wFmeasure': {'Task': [], 'Dataset': [], 'Value': []},
        'MAE': {'Task': [], 'Dataset': [], 'Value': []}
    }

    # 解析数据文件
    with open(data_file, 'r') as file:
        for line in file:
            try:
                parts = line.strip().split('; ')
                task = parts[0].split(': ')[1]
                dataset = parts[1].split(': ')[1]
                smeasure = float(parts[2].split(': ')[1].strip(';'))
                meanEm = parts[3].split(': ')[1].strip(';')
                wFmeasure = float(parts[4].split(': ')[1].strip(';'))
                mae = float(parts[5].split(': ')[1].strip(';'))

                data_dict['Smeasure']['Task'].append(task)
                data_dict['Smeasure']['Dataset'].append(dataset)
                data_dict['Smeasure']['Value'].append(smeasure)

                data_dict['meanEm']['Task'].append(task)
                data_dict['meanEm']['Dataset'].append(dataset)
                data_dict['meanEm']['Value'].append(float(meanEm) if meanEm != '-' else None)

                data_dict['wFmeasure']['Task'].append(task)
                data_dict['wFmeasure']['Dataset'].append(dataset)
                data_dict['wFmeasure']['Value'].append(wFmeasure)

                data_dict['MAE']['Task'].append(task)
                data_dict['MAE']['Dataset'].append(dataset)
                data_dict['MAE']['Value'].append(mae)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line}. Error: {e}")
                continue

    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(16, 20), gridspec_kw={'hspace': 0.3, 'wspace': 0.5})

    # 定义颜色
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown',
              'pink', 'olive', 'cyan', 'lime', 'teal', 'lavender', 'maroon',
              'turquoise', 'gold', 'darkred','b', 'g', 'r']

    # 设置全局字体
    # plt.rcParams['font.family'] = 'Times New Roman'

    # 子图布局
    metrics = ['Smeasure', 'meanEm', 'wFmeasure', 'MAE']
    axes = axs.flatten()

    # 绘制每个指标的柱状图
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df = pd.DataFrame(data_dict[metric]).dropna()

        # 获取每个任务的唯一标签和位置
        tasks = df['Task'].unique()
        bar_width = 0.8 / len(tasks)  # 控制每个任务的柱宽
        positions = list(range(len(df['Dataset'].unique())))  # 柱的基础位置

        # 绘制每个任务的数据
        for i, task in enumerate(tasks):
            task_data = df[df['Task'] == task]
            color = colors[i % len(colors)]
            pos = [p + i * bar_width for p in positions]  # 每个任务的柱的位置偏移

            ax.bar(pos, task_data['Value'], width=bar_width, color=color, label=task)

        # 设置轴标签和标题
        ax.set_xlabel('Dataset', fontsize=40)
        ax.set_ylabel(f'{metric} Value', fontsize=40)
        ax.set_xticks([p + (len(tasks) - 1) * bar_width / 2 for p in positions])
        ax.set_xticklabels(df['Dataset'].unique())
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

        # 为 Smeasure、meanEm 和 wFmeasure 设置 y 轴范围
        if metric in ['Smeasure', 'meanEm', 'wFmeasure']:
            ax.set_ylim(0.75, 1)  # 增加上限以提供更多空间

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图表为图片文件
    plt.savefig(output_file)

    # 显示图表
    plt.show()

    print(f"Plot saved as {output_file}")
