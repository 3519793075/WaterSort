import numpy as np
import pandas as pd
from sklearn import tree
import os, re
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"C:\Windows\Fonts\STXINGKA.TTF", size=15)  # 导入华文行楷文件

# 图像切割及特征提取C:\Users\锦程\Desktop\学科\数据可视化\water images
path = './water images1/'


# 自定义获取图片名称函数
def getImgNames(path=path):
    '''
    获取指定路径中所有图片的名称
    :param path: 指定的路径
    :return: 名称列表
    '''
    filenames = os.listdir(path)
    # filenames = os.makedirs(path)
    imgNames = []
    for i in filenames:
        # if re.findall('^\d_\d+\.jpg$', i) != []:
        # if re.findall('\d_\d+\.jpg', i) != []:
        if re.findall('\d_\d+\.jpg', i):
            imgNames.append(i)
    return imgNames


# 自定义获取三阶颜色矩函数
def Var(data=None):
    '''
    获取给定像素值矩阵的三阶颜色矩
    :param data: 给定的像素值矩阵
    :return: 对应的三阶颜色矩
    '''
    x = np.mean((data - data.mean()) ** 3)
    return np.sign(x) * abs(x) ** (1 / 3)


# 批量处理图片数据
imgNames = getImgNames(path=path)  # 获取所有图片名称
n = len(imgNames)  # 图片张数
data = np.zeros([n, 9])  # 用来装样本自变量
labels = np.zeros([n])  # 用来放样本标签

for i in range(n):
    img = Image.open(path + imgNames[i])  # 读取图片
    M, N = img.size  # 图片像素的尺寸
    img = img.crop((M / 2 - 50, N / 2 - 50, M / 2 + 50, N / 2 + 50))  # 图片切割
    r, g, b = img.split()  # 将图片分割成三通道
    rd = np.asarray(r) / 255  # 转化成数组数据
    gd = np.asarray(g) / 255
    bd = np.asarray(b) / 255

    data[i, 0] = rd.mean()  # 一阶颜色矩
    data[i, 1] = gd.mean()
    data[i, 2] = bd.mean()

    data[i, 3] = rd.std()  # 二阶颜色矩
    data[i, 4] = gd.std()
    data[i, 5] = bd.std()

    data[i, 6] = Var(rd)  # 三阶颜色矩
    data[i, 7] = Var(gd)
    data[i, 8] = Var(bd)

    labels[i] = imgNames[i][0]  # 样本标签

from sklearn.model_selection import train_test_split

# 数据拆分,训练集、测试集
print(data.shape)
print(labels)
data_tr, data_te, label_tr, label_te = train_test_split(data, labels, test_size=0.2, random_state=10)

from sklearn.tree import DecisionTreeClassifier

# 模型训练
model = DecisionTreeClassifier(random_state=5).fit(data_tr, label_tr)

# 水质评价
from sklearn.metrics import confusion_matrix

# 模型预测
pre_te = model.predict(data_te)
# 混淆矩阵
cm_te = confusion_matrix(label_te, pre_te)
print(cm_te)

plt.subplot(1, 2, 1)
plt.scatter(label_te, pre_te, s=5, c='r', marker='o', linewidths=5)
plt.xlabel('真实标签', fontproperties=font_set)
plt.ylabel('预测标签', fontproperties=font_set)
plt.title('散点图', fontproperties=font_set)

plt.subplot(1, 2, 2)
label_te1 = pd.Series(label_te)  # 真实标签
pre_te1 = pd.Series(pre_te)  # 预测标签
corr1 = label_te1.corr(pre_te1, 'pearson')  # person参数检测真实标签是否合预测标签在同一条线上
corr2 = label_te1.corr(pre_te1, 'spearman')
corr3 = label_te1.corr(pre_te1, 'kendall')
corr = [corr1, corr2, corr3]
# plt.bar(range(len(corr)),corr)
name = ['Pearson', 'Spearman', 'Kendall']
x = range(len(name))
plt.bar(x, corr)
plt.xticks(rotation=20)
plt.xticks(x, name)
# plt.tick_params(labelsize=17) 
plt.show()
