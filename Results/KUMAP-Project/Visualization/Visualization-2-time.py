################################################################################
# 对实验运行时间进行可视化
################################################################################
# 导入模块
import pandas as pd
from Draw import DrawBars

if __name__ == "__main__":
    path = "../KUMAP-Results/KUMAP-time.xlsx"
    data = pd.read_excel(path, sheet_name="Mean", index_col=0, header=0)

    # 对数据集和算法名字进行标准化
    data.rename(columns={'bloodmnist': 'BloodMNIST'}, inplace=True)
    data.rename(columns={'dermamnist': 'DermaMNIST'}, inplace=True)
    data.rename(columns={'octmnist': 'OCTMNIST'}, inplace=True)
    data.rename(columns={'organamnist': 'OrganAMNIST'}, inplace=True)
    data.rename(columns={'organcmnist': 'OrganCMNIST'}, inplace=True)
    data.rename(columns={'organsmnist': 'OrganSMNIST'}, inplace=True)
    data.rename(columns={'pneumoniamnist': 'PneumoniaMNIST'}, inplace=True)
    data.rename(columns={'retinamnist': 'RetinaMNIST'}, inplace=True)

    data = data.transpose()

    data.rename(columns={'ITML': 'SITML'}, inplace=True)
    data.rename(columns={'PACMAP': 'PaCMAP'}, inplace=True)
    data.rename(columns={'TRIMAP': 'TriMAP'}, inplace=True)

    # 将数据转化成字典格式
    values_ = data.to_dict(orient='list')

    colors = (
        "#00008C", "#2234A8", "#4467C4", "#659BDF",
        "#87CEFB", "#B1DDFC", "#FFE8A8", "#FFE59C",
        "#F3CB8E", "#E7B280", "#DA9871", "#CE7E63",
    )

    FIGSIZE = (6, 16)

    DB = DrawBars(
        path = "figures/experiments",
        ylabel = 'Average Execution Time (s)',
        color=colors,
        fontsize=15,
        titlefontsize=20,
        axis_flag=True,
        axis_expty=False,
        legend_flag=False
    )
    DB.filename = "Experiment-time-1"
    DB.grouped_barhs(values=values_, groups=data.index, ylim=(0, 150), figsize=FIGSIZE)

    DB.legend_flag = True
    DB.filename = "Experiment-time-2"
    DB.grouped_barhs(values=values_, groups=data.index, ylim=(200, 400), figsize=FIGSIZE)
