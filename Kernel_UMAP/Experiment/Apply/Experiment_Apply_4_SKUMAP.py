################################################################################
# 实验二：SKUMAP分类效果实验
################################################################################
# 导入模块
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
from Factory import factory
from Assess import Analysis_Kernel_Manifold
from KUMAP import KUMAP_config
config = KUMAP_config()
################################################################################
for ds in config.KUMAP_data:
    model = factory(
        func_name='SKUMAP',
        data_name=ds,
        return_time=config.return_time,
        train_size=config.train_size,
        random_state=config.random_state,
        sec_part='Experiment-Apply',
        sec_num=4
    )
    model.Product_Kernel_Object(
        train_func='UMAP',
        Kernel_func=config.Kernel_func,
        ALpha=config.Alpha,
        is_pca=config.is_pca,
        pca_n=config.pca_n,
        is_Supervised=True,
        label=None,
        label_num=config.label_num,
        ita=config.ita
    )
    model.Kernel_Object.fit_transform()
    Analysis_Kernel_Manifold(model.Kernel_Object).Analysis()
