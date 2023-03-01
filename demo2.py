import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sn
from sklearn.metrics import confusion_matrix
plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False


# # 加载数据，看看目录下的图片
# imgFile = './images/1_1.jpg'
# img = Image.open(imgFile)
# print("图片大小为:", img.size)
# plt.figure(figsize=(10, 8))
# plt.imshow(img)
# plt.title("水色样本:" + imgFile + "类别标签:" + str(imgFile[9]))
#
# # 不显示坐标轴
# plt.axis('off')
# plt.show()


# 开始对图像的颜色矩特征进行提取

# 通过对数据图像的观察，决定选取每张图像中心处的图像进行颜色矩特征提取

def img_extract():
    input_path = './water images1'
    output_path = '../小方测试/result.csv'
    result = []
    imglist = os.listdir(input_path)
    # print(imglist) 得到类似于1_1.jpg这样的文件名,其中1_1表示属于1类别的第一张图片
    for i in range(len(imglist)):
        # 开始把类别和第几张图片分开
        num = imglist[i].rstrip('.jpg').split('_')
        # print(num)  # 得到['1','1']这样的列表
        # 把字符串列表转换为数值型
        num = [int(x) for x in num]

        # 开始图像分割
        img = Image.open(input_path + '/' + imglist[i])
        h, w = img.size
        # 取图片中心100*100的图像
        # 关于crop的介绍:https://blog.csdn.net/banxia1995/article/details/85330212
        box = [h / 2 - 50, w / 2 - 50, h / 2 + 50, w / 2 + 50]
        small = img.crop(box)

        # 提取颜色特征
        rgb = np.array(small) / [255.0, 255.0, 255.0]
        # print(rgb)
        # 一阶颜色矩
        first_order = 1.0 * (rgb.sum(axis=0).sum(axis=0)) / 10000
        err = rgb - first_order
        # print(first_order)

        # 二阶颜色矩
        second_order = np.sqrt(1.0 * (np.power(err, 2)).sum(axis=0).sum(axis=0) / 10000)

        # 三阶颜色矩
        third_order = 1.0 * (pow(err, 3).sum(axis=0).sum(axis=0)) / 10000
        third_order = np.cbrt(abs(third_order)) * -1.0
        # print(third_order)

        res = np.concatenate((num, first_order, second_order, third_order))
        result.append(res)

    # 保存到csv文件G
    names = ['水质类别', '序号', 'R通道一阶矩', 'G通道一阶矩', 'B通道一阶矩',
             'R通道二阶矩', 'G通道二阶矩', 'B通道二阶矩',
             'R通道三阶矩', 'G通道三阶矩', 'B通道三阶矩']

    df = pd.DataFrame(result, columns=names)
    # print(df)
    df.to_csv(output_path, encoding='utf-8', index=False)


# img_extract()

# 开始导入我们做好的特征数据集，进行数据集划分并建模
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('../小方测试/result.csv', encoding='utf-8')
data.head()

# 因为数据很小，为了增大数据区分度所以对X乘以30，避免过拟合
X = data.iloc[:, 2:] * 30
Y = data['水质类别']
Y = Y.astype(int) - 1
Y[Y > 0] = 1
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# print(X_train.shape, X_test.shape)

# 神经网络
import tensorflow as tf
# from keras.utils import np_utils
# from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Activation

# reshape
# X_train = X_train.astype('float32').values.reshape(len(X_train), 3, 3, 1)
# X_test = X_test.astype('float32').values.reshape(len(X_test), 3, 3, 1)

# 因为类别不是从0开始编号，所以进行one-hot编码时减1
# Y_train = np_utils.to_categorical(y_train - 1)
# Y_test = np_utils.to_categorical(y_test - 1)

# 参数设置
input_size = 4
hidden_size = 8
output_size = 3
learning_rate = 0.01

tf.keras.backend.clear_session()
# model = Sequential()
# model.add(tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'))
# model.add(tf.keras.layers.Dense(output_size, activation=tf.keras.activations.softmax))
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate), loss=tf.keras.losses.categorical_crossentropy,
#               metrics=['accuracy'])
# interpretation_train = tf.squeeze(Y_train)
model = tf.keras.Sequential() # 建立模型
model.add(Dense(input_dim=9,units =9))
model.add(Activation('relu'))
model.add(Dense(input_dim=9,units =1))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
tf.data.experimental.enable_debug_mode()# 启动调试模式

# 模型训练
# hist = model.fit(log_train, tf.one_hot(interpretation_train, depth=output_size), epochs=2000)

print('Training')
result = model.fit(X_train, y_train, epochs=40, batch_size=6, validation_split=0.1, shuffle=True)

print('Testing')
loss, accuracy = model.evaluate(X_test, y_test)
print('loss, accuracy', loss, accuracy)

N = 40
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), result.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), result.history["val_accuracy"], label="val_acc")
plt.title("loss and accuracy")
plt.xlabel("epoch")
plt.ylabel("loss/acc")
plt.legend(loc="best")
plt.show()

cnn_pred = model.predict(X_test)
cnn_pred = np.argmax(cnn_pred, axis=1) + 1

cnn_testcm = metrics.confusion_matrix(y_test, cnn_pred)
confusion_matrix(cnn_testcm)
