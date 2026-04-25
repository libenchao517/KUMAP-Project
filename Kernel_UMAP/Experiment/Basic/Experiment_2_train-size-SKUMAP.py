################################################################################
# 实验二：数据划分实验
################################################################################
# 加入路径
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入模块
from Assess import Analysis_Kernel_Paras
from Draw import Draw_Line_Chart
from Factory import factory
from KUMAP import KUMAP_config
################################################################################
# 初始化变量
config = KUMAP_config()
paraname = "train_size"
train_size = [0.01] + [0.01 * (i+1) for i in range(4, 99, 5)]
################################################################################
# 定义模型
for ds in config.basic_data:
    model = factory(
        func_name='SKUMAP',
        data_name=ds,
        return_time=True,
        train_size=None,
        random_state=config.random_state,
        sec_part='Experiment-Basic',
        sec_num=2)
# 生产对象
    model.Product_Kernel_Object(
        train_func='UMAP',
        Kernel_func=config.Kernel_func,
        ALpha=config.Alpha,
        is_pca=config.is_pca,
        pca_n=config.pca_n,
        is_Supervised=True,
        label=None,
        label_num=2,
        ita=0.5)
    AKSS = Analysis_Kernel_Paras(object=model.Kernel_Object, para=train_size, paraname=paraname)
################################################################################
# 运行实验
    AKSS.Analysis()
################################################################################
# 绘制结果
    file = model.sec_part + "-" + str(model.sec_num)+ "-" + ds + "-" + model.func_name + "-" + paraname
    AKSS.data_name = ds

    Draw_Line_Chart(
        filename=file,
        column=train_size,
        left=[AKSS.KNN, AKSS.PRE, AKSS.REC, AKSS.F1],
        right=[AKSS.time],
        ylim_left=(0, 1.05),
        left_marker=(["^", "v", "<", ">"]),
        right_marker=(["o",]),
        left_color=("#71BFB2", "#FEE066", "#237B9F", "#AD0B08"),
        right_color=("#EC817E",),
        left_markeredgecolor=("#71BFB2", "#FEE066", "#237B9F", "#AD0B08"),
        right_markeredgecolor=("#EC817E",),
        left_markerfacecolor=("#71BFB2", "#FEE066", "#237B9F", "#AD0B08"),
        right_markerfacecolor=("#EC817E",),
        ylabel_left="Mean Classification Performance",
        ylabel_right="Time Costs",
        xlabel="Size of Train Set (%)",
        # fontsize=18,
        # titlefontsize=20,
        left_label=["Accuracy", "Precision", "Recall", "F1 Score"],
        right_label=["Time (s)", ],
    ).Draw_double_line()

    analysis = Path("Analysis")
    analysis.mkdir(exist_ok=True)
    AKSS.reaults.to_excel(analysis.joinpath(file + ".xlsx"))
