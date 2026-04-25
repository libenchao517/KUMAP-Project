################################################################################
# 对新标签向量维度和信息比例进行可视化
################################################################################
# 导入模块
import os
import pandas as pd
from Draw import Draw_Line_Chart
from KUMAP import KUMAP_config

if __name__ == "__main__":
    config = KUMAP_config()
    path = "../KUMAP-Pre-Test-2026-03-28-07-28/Analysis"
    xlabels = [
        "Dimensions $\eta$ of Label Vectors",
        "Proportion $\lambda$ of Data Information"
    ]

    for dn in config.basic_data:
        for idx, para in enumerate(["label_num", "ita"]):
            filename = "-".join(["Experiment", "Basic", str(idx+6), dn, "SKUMAP", para])
            xlsx_path = os.path.join(path, filename + ".xlsx")
            data = pd.read_excel(xlsx_path, index_col=0, header=0)
            Draw_Line_Chart(
                path="figures/parameters",
                filename=filename,
                column=data.index,
                left=[data.KNN, data.PRE, data.REC, data.F1],
                right=[data.time],
                ylim_left=(0, 1.05),
                ylim_right=None,
                left_marker=(["^", "v", "<", ">"]),
                left_color=("#71BFB2", "#FEE066", "#237B9F", "#AD0B08"),
                left_markeredgecolor=("#71BFB2", "#FEE066", "#237B9F", "#AD0B08"),
                left_markerfacecolor=("#71BFB2", "#FEE066", "#237B9F", "#AD0B08"),
                ylabel_left="Classification Performance",
                xlabel=xlabels[idx],
                fontsize=15,
                titlefontsize=18,
                left_label=["Accuracy", "Precision", "Recall", "F1 Score"],
            ).Draw_simple_line()
