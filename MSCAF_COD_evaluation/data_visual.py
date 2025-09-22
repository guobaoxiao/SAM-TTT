import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker
# 读取数据文件
data_file = 'result.txt'
output_file = 'evaluation_metrics_comparison.png'


def datavisual(data_file, output_file):
    # 检查文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"The data file {data_file} does not exist.")

    # 初始化字典以存储数据
    data_dict = {
        'Smeasure': {'Task': [], 'Dataset': [], 'Value': []},
        'meanEm': {'Task': [], 'Dataset': [], 'Value': []},
        # 'maxEm': {'Task': [], 'Dataset': [], 'Value': []},
        'wFmeasure': {'Task': [], 'Dataset': [], 'Value': []},
        # 'meanFm': {'Task': [], 'Dataset': [], 'Value': []},
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
                # maxEm = parts[4].split(': ')[1].strip(';')
                wFmeasure = float(parts[4].split(': ')[1].strip(';'))
                # meanFm = float(parts[6].split(': ')[1].strip(';'))
                mae = float(parts[5].split(': ')[1].strip(';'))

                data_dict['Smeasure']['Task'].append(task)
                data_dict['Smeasure']['Dataset'].append(dataset)
                data_dict['Smeasure']['Value'].append(smeasure)

                data_dict['meanEm']['Task'].append(task)
                data_dict['meanEm']['Dataset'].append(dataset)
                data_dict['meanEm']['Value'].append(float(meanEm) if meanEm != '-' else None)

                # data_dict['maxEm']['Task'].append(task)
                # data_dict['maxEm']['Dataset'].append(dataset)
                # data_dict['maxEm']['Value'].append(float(maxEm) if maxEm != '-' else None)

                data_dict['wFmeasure']['Task'].append(task)
                data_dict['wFmeasure']['Dataset'].append(dataset)
                data_dict['wFmeasure']['Value'].append(wFmeasure)

                # data_dict['meanFm']['Task'].append(task)
                # data_dict['meanFm']['Dataset'].append(dataset)
                # data_dict['meanFm']['Value'].append(meanFm)

                data_dict['MAE']['Task'].append(task)
                data_dict['MAE']['Dataset'].append(dataset)
                data_dict['MAE']['Value'].append(mae)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line}. Error: {e}")
                continue

    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(15, 17))
    # fig.suptitle('Evaluation Metrics Comparison Across Tasks and Datasets', fontsize=16)

    # 定义颜色和标记样式
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown',
              'pink', 'olive', 'cyan', 'lime', 'teal', 'lavender', 'maroon',
              'turquoise', 'gold', 'darkred']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+', '*', 'p', 'h', '|', '_',
               '.', ',', '1', '2', '3']

    # 子图布局
    metrics = ['Smeasure', 'meanEm', 'wFmeasure', 'MAE']
    # metrics = ['Smeasure', 'meanEm', 'maxEm', 'wFmeasure', 'meanFm', 'MAE']
    axes = axs.flatten()

    # 绘制每个指标的子图
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df = pd.DataFrame(data_dict[metric]).dropna()
        for i, task in enumerate(df['Task'].unique()):
            task_data = df[df['Task'] == task]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            ax.plot(task_data['Dataset'], task_data['Value'], marker=marker, color=color, label=task, markerfacecolor='none', linewidth=3, markersize=12, markeredgewidth=2)

        # ax.set_title(f'{metric}', fontsize=25)  # 设置标题字号
        ax.set_xlabel('Dataset', fontsize=40)  # 设置x轴标签字号
        ax.set_ylabel(f'{metric} Value', fontsize=40)  # 设置y轴标签字号
        ax.legend(fontsize=25)  # 设置图例字号
        ax.tick_params(axis='both', which='major', labelsize=20)  # 设置坐标轴刻度标签字号
        # for spine in ax.spines.values():
        #     spine.set_linewidth(5)  # 将边框线条加粗
        # 为刻度标签加粗
        # for tick in ax.get_xticklabels():
        #     tick.set_fontweight('bold')
        # for tick in ax.get_yticklabels():
        #     tick.set_fontweight('bold')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  # Y轴显示三位小数

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图表为图片文件
    plt.savefig(output_file)

    # 显示图表
    plt.show()

    print(f"Plot saved as {output_file}")

# 调用函数
datavisual(data_file, output_file)
