################################################################################
# 对比实验四：PACMAP实验
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
        func_name='PACMAP',
        data_name=ds,
        return_time=config.return_time,
        train_size=None,
        random_state=config.random_state,
        sec_part='Comparatation-Total',
        sec_num=4
    )
    # 生产对象
    model.Product_Standard_Object(func_type='')
    # 运行模型
    model.Standard_Object.Embedded()
    # 结果评价
    Analysis_Standard_Contrast_Manifold(model.Standard_Object).Analysis()
