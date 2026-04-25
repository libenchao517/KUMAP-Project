################################################################################
# 实验三：分布观察实验
################################################################################
# 加入路径
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入模块
from Draw import Draw_Pairhist
from Factory import factory
from KUMAP import KUMAP_config
from sklearn.metrics import pairwise_distances
config = KUMAP_config()
################################################################################
# 定义模型
for ds in config.basic_data:
    model = factory(
        func_name='KUMAP',
        data_name=ds,
        return_time=False,
        train_size=0.5,
        random_state=config.random_state,
        sec_part='Experiment-Basic',
        sec_num=3
    )
# 生产对象
    model.Product_Kernel_Object(
        train_func='UMAP',
        Kernel_func=config.Kernel_func,
        ALpha=config.Alpha,
        is_pca=config.is_pca,
        pca_n=config.pca_n,
        is_Supervised=False,
        label=None,
        label_num=None,
        ita=None
    )
################################################################################
# 运行实验
    model.Kernel_Object.fit_transform()
################################################################################
# 计算距离矩阵
    dist_train = pairwise_distances(model.Kernel_Object.Y_train)
    dist_test = pairwise_distances(model.Kernel_Object.Y_test)
################################################################################
# 结果可视化
    Dh = Draw_Pairhist(fontsize=15, titlefontsize=18)
    Dh.filename = model.sec_part + "-" + str(model.sec_num)+ "-" + ds + "-KUMAP-distance"
    Dh.Draw_pairhist(dist_train, dist_test)
