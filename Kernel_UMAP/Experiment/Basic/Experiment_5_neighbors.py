################################################################################
# 实验五：邻居数分析实验
################################################################################
# 导入模块
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
import numpy as np
from Assess import K_Nearest_Neighbors
from Factory import factory
from KUMAP import KUMAP_config
from Draw import Draw_Line_Chart
from matplotlib import rcParams
rcParams['figure.autolayout'] = True
rcParams.update({'font.size': 18})
################################################################################
# 初始化变量
config = KUMAP_config()
N = 20
odds = np.zeros((2, N))
################################################################################
# 初始化方法
methods = {
    'KUMAP':
        {
            'is_pca': config.is_pca,
            'pca_n': config.pca_n,
            'is_Supervised': False,
            'label': None,
            'label_num': None,
            'ita': None,
            'train_size': config.train_size,
            'Alpha': config.Alpha,
            'train_func': 'UMAP',
            'Kernel_func': config.Kernel_func,
        },
    'SKUMAP': {
        'is_pca': config.is_pca,
        'pca_n': config.pca_n,
        'is_Supervised': True,
        'label': None,
        'label_num': config.label_num,
        'ita': config.ita,
        'train_size': config.train_size,
        'Alpha': config.Alpha,
        'train_func': 'UMAP',
        'Kernel_func': config.Kernel_func,
    }
}
################################################################################
for ds in config.basic_data:
    for j, m in enumerate(methods.keys()):
        # 定义模型
        model = factory(
            func_name=m,
            data_name=ds,
            return_time=True,
            random_state=config.random_state,
            train_size=methods.get(m).get('train_size'),
            sec_part='Experiment-Basic',
            sec_num=5
        )
        # 生产对象
        model.Product_Kernel_Object(
            train_func=methods.get(m).get('train_func'),
            Kernel_func=methods.get(m).get('Kernel_func'),
            ALpha=methods.get(m).get('Alpha'),
            is_pca=methods.get(m).get('is_pca'),
            pca_n=methods.get(m).get('pca_n'),
            is_Supervised=methods.get(m).get('is_Supervised'),
            label=methods.get(m).get('label'),
            label_num=methods.get(m).get('label_num'),
            ita=methods.get(m).get('ita')
        )

        # 运行实验
        model.Kernel_Object.fit_transform()

        for i in range(N):
            knn = K_Nearest_Neighbors(neighbors=i + 1)
            knn.KNN_predict_odds_splited(model.Kernel_Object.Y_train, model.Kernel_Object.Y_test, model.Kernel_Object.T_train, model.Kernel_Object.T_test, model.Kernel_Object.para)
            odds[j, i] = knn.accuracy

    Draw_Line_Chart(
        filename=model.sec_part+"-"+str(model.sec_num)+"-"+ds+"-neighbors",
        left=odds,
        left_color=["#336872", "#EF7B30"],
        xlabel='Neighbors $k$',
        ylabel_left='Accuracy',
        ylim_left=(0.0, 1.05),
        left_label=["KUMAP", "SKUMAP"],
        left_marker=["+", "x"],
        fontsize=15,
        titlefontsize=18
    ).Draw_simple_line()
