import glob

import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = layers.Dense(units=512, activation='relu')
        self.out = layers.Dense(units=10, activation='softmax')

    def call(self, input):
        x1 = self.linear(input)
        x2 = self.out(x1)
        # out = tf.argmax(x2, axis=1)
        # return out
        return x2

def read_image(path_list):
    '''
    :param path_list:文件的路径list
    :return: 图片张量列表,图片标签列表
    '''
    images = []
    for i in path_list:
        print(i)
        img_temp = tf.io.read_file(i)
        # 手写数字数据集是黑白的一通道灰度图
        img_temp = tf.image.decode_jpeg(img_temp, channels=1)
        # 打平为一维
        img_temp = tf.image.resize(img_temp, [1, 784])
        images.append(img_temp)
    return np.array(images)

if __name__ == '__main__':
    train_path = "./flag/"
    # 这个函数用于返回符合,可以使用正则路径，*表示任意字符
    # path_list = tf.data.Dataset.list_files(train_path + "*.jpg")
    path_list=sorted(glob.glob(train_path+"*.jpg"))
    # 读取训练图片
    train_images = read_image(path_list)
    train_images /= 255

    model = Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.build(input_shape=[None, 784])
    model.summary()

    checkpoint = tf.train.Checkpoint(model)
    checkpoint_path = './weights/my_save_weights'
    checkpoint.restore(checkpoint_path)

    for item in train_images:
        pre = model(item, training=False)
        pre = tf.argmax(pre, axis=1)
        print(pre)

