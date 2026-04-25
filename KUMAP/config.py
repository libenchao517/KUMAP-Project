################################################################################
# 本文件用于存储Kernel UMAP算法的关键参数
################################################################################
# 导入模块
import numpy as np
################################################################################
class KUMAP_config:
    """
    定义Kernel UMAP算法的关键参数
    """
    def __init__(self):
        # 初始化函数
        self.return_time = True            # 是否返回时间
        self.Kernel_func = 'Exponential'   # 核函数
        self.train_size = 0.20             # 训练数据比例
        self.random_state = np.random.randint(2024) # 随机种子
        self.Alpha = 0.080                 # 核函数的超参数
        self.is_pca = True                 # 是否使用PCA进行预处理
        self.pca_n = 30                    # PCA预处理的维度
        self.label_num = 8                 # 新标签的维度
        self.ita = 0.20                    # SKUMAP在的超参数
        self.none_data=[]
        self.KUMAP_data = [                # KUMAP使用的数据集
            "bloodmnist",
            "dermamnist",
            "octmnist",
            "organamnist",
            "organcmnist",
            "organsmnist",
            "pneumoniamnist",
            "retinamnist"
        ]
        self.basic_data = [                # 基本参数实验使用的数据集
            "bloodmnist",
            "organcmnist",
            "organsmnist",
            "pneumoniamnist"
        ]
        self.KUMAP_select_target = {}
