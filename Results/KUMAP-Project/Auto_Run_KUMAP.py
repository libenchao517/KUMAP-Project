################################################################################
# 本代码用于自动化运行KUMAP项目实验
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入模块
from Utils import Auto_Run
################################################################################
# 运行项目
AR = Auto_Run(
    Project="KUMAP",
    MRPY="Make_Pre_Results_KUMAP.py",
    content="Kernel_UMAP/Experiment/Basic",
    lock=True
)
AR.Run()

for i in range(10):
    AR = Auto_Run(
        Project="KUMAP",
        MRPY=None,
        content="Kernel_UMAP/Comparatation",
        lock=True
    )
    AR.Run()

    AR = Auto_Run(
        Project="KUMAP",
        MRPY="Make_Results_KUMAP.py",
        content="Kernel_UMAP/Experiment/Apply",
        lock=True
    )
    AR.Run()

AR = Auto_Run(
    Project="KUMAP",
    MRPY=None,
    content="Results/KUMAP-Project",
    run_file="Total_KUMAP.py",
    is_parallel=False,
    lock=True
)
AR.Run()
