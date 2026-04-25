################################################################################
# 对比实验一：UMAP实验
################################################################################
# 导入模块
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
from Factory import factory
from Assess import Analysis_Standard_Contrast_Manifold
from KUMAP import KUMAP_config
config = KUMAP_config()
################################################################################
for ds in config.KUMAP_data:
    # 定义模型
    model = factory(
        func_name='UMAP',
        data_name=ds,
        return_time=config.return_time,
        random_state=config.random_state,
        train_size=None,
        sec_part='Comparatation-Total',
        sec_num=1
    )
    # 生产对象
    model.Product_Standard_Object(func_type='')
    # 运行模型
    model.Standard_Object.Embedded()
    # 结果评价
    Analysis_Standard_Contrast_Manifold(model.Standard_Object).Analysis()
