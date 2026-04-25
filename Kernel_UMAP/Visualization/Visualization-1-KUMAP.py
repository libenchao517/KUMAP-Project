################################################################################
# 可视化数据集
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入模块
import os
import numpy as np
from PIL import Image
from DATA import Load_Data
from KUMAP import KUMAP_config
################################################################################
# 绘制单张图片
config = KUMAP_config()
os.makedirs("KUMAP-Figure", exist_ok=True)
for dn in config.KUMAP_data:
    os.makedirs("KUMAP-Figure/" + dn, exist_ok=True)
    data, target = Load_Data(dn, is_scaler=False).Load_MedMNIST()
    mode = ""
    if data.shape[1] == 784:
        data = data.reshape((data.shape[0], 28, 28))
        mode = "L"
    else:
        data = data.reshape((data.shape[0], 28, 28, 3))
        mode = "RGB"
    cates, count = np.unique(target, return_counts=True)
    for cate in cates:
        temp = data[target == cate]
        for i in range(5):
            image = Image.fromarray(temp[i], mode)
            image.save("KUMAP-Figure/" + dn + "/" +dn+"-"+str(cate)+"-"+str(i)+".png")
################################################################################
# 组合图片
root_path = "KUMAP-Figure"
os.makedirs("Figure", exist_ok=True)
os.makedirs("Figure/KUMAP-Figure", exist_ok=True)
name_list = [item for item in os.listdir(root_path)]
for name in name_list:
    # img = Image.new("RGB", (280, 28), color=(255, 255, 255))
    img = Image.new("RGB", (5000, 500), color=(255, 255, 255))
    data_path = os.path.join(root_path, name)
    file_list = os.listdir(data_path)
    for i, file in enumerate(file_list):
        file_path = os.path.join(data_path, file)
        image = Image.open(file_path)
        image = image.resize((500, 500))
        # img.paste(image, (i*28, 0))
        img.paste(image, (i*500, 0))
    img.save(os.path.join("Figure", root_path, name + ".png"))
