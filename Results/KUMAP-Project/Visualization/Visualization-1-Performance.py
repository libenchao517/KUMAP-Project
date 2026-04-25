################################################################################
# 本代码用于可视化KUMAP实验结果
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入必要模块
import os
import numpy as np
import pandas as pd
from Draw import Error_Drawing
from Utils import Make_Table
################################################################################
# 定义基本变量
project = "KUMAP"
index = ["KNN", "PRE", "REC", "F1", "time", "CPU", "GPU"]
index_p = ["KNN", "PRE", "REC", "F1"]

methods = [
    'LDA','KLDA',
    'PACMAP', 'TRIMAP', 'UMATO', 'MDE', 'UMAP',
    'NCA', 'ITML',
    'PUMAP', 'KUMAP','SKUMAP']

# 按照methods中的顺序进行调整
MT = Make_Table(methods=methods)
for idx in index:
    MT.Make(os.path.join("..", "KUMAP-Results", project+ '-' + idx + '.xlsx'))

# 设置颜色
colors = [[
    "#FFFF99", "#FEDFB8", "#FEA042", "#FB7E00",
    "#E31A1C", "#FA9B99", "#CBBED5", "#6A3D9A",
    "#1D78B5", "#A6CEE3", "#32A02C", "#B2E08A"
], ]

for i, idx in enumerate(index_p):
    mean = pd.read_excel(os.path.join("../KUMAP-Results", project + '-' + idx + '.xlsx'), sheet_name="Mean", index_col=0, header=0)
    std = pd.read_excel(os.path.join("../KUMAP-Results", project + '-' + idx + '.xlsx'), sheet_name="Std", index_col=0, header=0)

    ED = Error_Drawing(
        path='figures/experiments',
        filename=None,
        fontsize=15,
        titlefontsize=20,
        title=None,
        xlabel=None,
        ylabel=None,
        formats=("png", )
    )
    for j, data in enumerate(mean.columns):
        mean_ = mean[data]
        std_ = std[data]
        ED.filename = '-'.join([idx, data])
        ED.drawing_bar_error(
            x_value = methods,
            mean_value = mean_,
            std_value = std_,
            colors=np.array(colors),
            labels=None,
            legend_flag=False,
            ylim_flag=True,
            axis_flag=False,
        )
