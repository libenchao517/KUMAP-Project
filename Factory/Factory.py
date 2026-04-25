################################################################################
# 本文件用于实现流形学习算法模型工厂
################################################################################
# 导入模块
import numpy as np
import random
import datetime
import platform
from sklearn.decomposition import PCA
from Contrast import Contrast_Method_KUMAP
from DATA import Load_Data
from DATA import datas
from DATA import abbre_labels
from DATA import Pre_Procession as PP
from KUMAP import KUMAP
from KUMAP import KUMAP_Supervised_Labels
from KUMAP import KUMAP_config
################################################################################
class factory:
    """
    模型工厂
    """
    def __init__(
        self,
        func_name='UMAP',
        data_name='USPS',
        return_time=True,
        train_size=0.1,
        split_type=None,
        random_state=None,
        is_noisy=False,
        sigma=0.01,
        is_clip=False,
        clip_num=0,
        is_select_target = False,
        target_num = 0,
        sec_part='Comparatation',
        sec_num=0,
    ):
        """
        初始化函数
        :param func_name:   算法名称
        :param data_name:   数据名称
        :param return_time: 是否返回时间
        :param train_size:  训练比例
        :param split_type:  训练集测试集划分方法
        :param random_state: 随机种子
        :param is_noisy:     是否添加噪声
        :param sigma:        高斯噪声强度
        :param is_clip:      是否切割数据
        :param clip_num:     子集的规模
        :param is_select_target: 是否根据类别进行采样
        :param target_num:   选择的类别的数量
        :param sec_part:     项目名称
        :param sec_num:      实验序号
        """
        self.xn = 80
        print("#" * self.xn)
        print(func_name + "算法性能测试")
        print("*" * self.xn)
        print("性能指标：")
        print("*" * self.xn)
        print("测试日期：", datetime.date.today())
        print("测试时间：", datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("计算机名：", platform.node())
        print("操作系统：", platform.system())
        print("解 释 器：", platform.python_version())
        print("数 据 集：", data_name)
        print("算法名称：", func_name)
        print("*" * self.xn)
        self.data_name = data_name
        self.func_name = func_name
        self.random_state = random_state
        self.return_time = return_time
        self.train_size = train_size
        self.split_type = split_type
        self.is_noisy = is_noisy
        self.sigma =sigma
        self.is_clip = is_clip
        self.clip_num = clip_num
        self.is_select_target = is_select_target
        self.target_num = target_num
        self.sec_part = sec_part
        self.sec_num = sec_num

    def Product_Standard_Object(self, func_type=''):
        """
        KUMAP项目中生成对比算法的工厂
        :param func_type: PUMAP的类型
        :return: Standard_Object
        """
        config = KUMAP_config()
        # 加载数据集
        if self.data_name in datas.get("MEDMNIST_2D") or self.data_name in datas.get("MEDMNIST_3D"):
            data, target = Load_Data(self.data_name).Load_MedMNIST()
        else:
            data, target = Load_Data(self.data_name).Loading()
        legd = abbre_labels.get(self.data_name)
        # 对数据集的类别的进行随机采样
        if self.is_select_target:
            index_list = [i for i in range(len(legd))]
            index_select = random.sample(index_list, k=self.target_num)
            index_select.sort()
            config.KUMAP_select_target[self.data_name] = index_select

        if config.KUMAP_select_target.get(self.data_name) is not None:
            data, target = PP().select_target(data, target, config.KUMAP_select_target.get(self.data_name))

        target, mapping = PP().target_mirror(target)
        if config.KUMAP_select_target.get(self.data_name) is not None and legd is not None:
            temp = [int(mapping.get(i)) for i in config.KUMAP_select_target.get(self.data_name)]
            legd = [legd[i] for i in temp]
        # 采用数据集的子集
        if self.is_clip:
            _, data, _, target = PP().sub_one_sampling(data, target, train_size=self.clip_num, random_state=self.random_state)
        # 为数据添加高斯噪声
        if self.is_noisy:
            data = PP().add_gaussian_noise(data, sigma=self.sigma)

        print("*" * self.xn)
        # 初始化对象
        self.Standard_Object = Contrast_Method_KUMAP(
            data=data,
            target=target,
            func_name=self.func_name,
            func_type=func_type,
            data_name=self.data_name,
            train_size=self.train_size,
            random_state=self.random_state,
            return_time=self.return_time,
            sec_part=self.sec_part,
            sec_num=self.sec_num)
        self.Standard_Object.legd = legd
        return self.Standard_Object

    def Product_Kernel_Object(
            self,
            train_func='UMAP',
            Kernel_func='Gauss',
            ALpha=0.09,
            is_pca=False,
            pca_n=30,
            is_Supervised=False,
            label=None,
            label_num=None,
            ita=None
    ):
        """
        生产核化流形学习方法的工厂
        :param train_func:   基本流形学习方法
        :param Kernel_func:  核函数
        :param ALpha:        核函数的超参数
        :param is_pca:       是否使用PCA进行预处理
        :param pca_n:        PCA预处理的目标维度
        :param is_Supervised: 是否是有监督的方法
        :param label:        新的标签向量
        :param label_num:    新标签向量的维度
        :param ita:          数据和标签的重要性
        :return:
        """
        config = KUMAP_config()
        # 加载数据集
        if self.data_name in datas.get("MEDMNIST_2D") or self.data_name in datas.get("MEDMNIST_3D"):
            data, target = Load_Data(self.data_name).Load_MedMNIST()
        else:
            data, target = Load_Data(self.data_name).Loading()
        legd = abbre_labels.get(self.data_name)
        # 对数据集的类别的进行随机采样
        if self.is_select_target:
            index_list = [i for i in range(len(legd))]
            index_select = random.sample(index_list, k=self.target_num)
            index_select.sort()
            config.KUMAP_select_target[self.data_name] = index_select

        if config.KUMAP_select_target.get(self.data_name) is not None:
            data, target = PP().select_target(data, target, config.KUMAP_select_target.get(self.data_name))

        target, mapping = PP().target_mirror(target)
        if config.KUMAP_select_target.get(self.data_name) is not None and legd is not None:
            temp = [int(mapping.get(i)) for i in config.KUMAP_select_target.get(self.data_name)]
            legd = [legd[i] for i in temp]
        # 采用数据集的子集
        if self.is_clip:
            _, data, _, target = PP().sub_one_sampling(data, target, train_size=self.clip_num, random_state=self.random_state)
        # 为数据集添加高斯噪声
        if self.is_noisy:
            data = PP().add_gaussian_noise(data, sigma=self.sigma)

        print("*" * self.xn)
        # 使用PCA对数据集进行预处理
        if is_pca and data.shape[1] >= pca_n:
            if pca_n >= data.shape[0]:
                pca_n = data.shape[0] - 1
            PCA_out = PCA(n_components=pca_n).fit_transform(data)
        else:
            is_pca = False
        # 生成新的标签向量
        if is_Supervised:
            label = KUMAP_Supervised_Labels().Make_label(data, target, label_n=label_num, target_n=len(set(target)))
        # 划分训练集和测试集
        train, test = None, None
        if self.split_type == "uniform":
            train, test = PP().uniform_sampling_index(data, target, train_size=self.train_size, random_state=self.random_state)
        # 初始化对象
        self.Kernel_Object = KUMAP(
            data=PCA_out if is_pca else data,
            target=target,
            func_name=self.func_name,
            data_name=self.data_name,
            train_func=train_func,
            Kernel_func=Kernel_func,
            is_Supervised=is_Supervised,
            label=label,
            label_num=label_num,
            ita=ita,
            train_size=self.train_size,
            random_state=self.random_state,
            train_index=train,
            test_index=test,
            ALpha=ALpha,
            return_time=self.return_time,
            sec_part=self.sec_part,
            sec_num=self.sec_num)
        self.Kernel_Object.legd = legd
        return self.Kernel_Object
