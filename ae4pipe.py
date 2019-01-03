import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import os.path

import time
from time import sleep
from datetime import datetime

import chainer
from chainer import cuda
from chainer import datasets
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import serializers
from chainer import training
from chainer.training import extensions

import glob
from itertools import chain

from PIL import Image
from PIL import ImageOps


width, height = 128, 74
hidden = 2048
epoch_num = 100
batchsize = 128
gpu_id = 0


class Autoencoder(chainer.Chain):
    def __init__(self, n_in, n_units):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_in)

    def __call__(self, x):
        y = self.forward(x)
        loss = F.mean_squared_error(x, y)
        return loss

    def forward(self, x):
        h = F.sigmoid(self.l1(x))
        y = F.sigmoid(self.l2(h))
        return y


class ResizedImageDataset(object):
    def __init__(self, path, size):
        self.path = path
        self.size = size

    def load_images_as_dataset(self):
        images_path_list = glob.glob('{}/*.jpg'.format(self.path))
        dataset = chainer.datasets.ImageDataset(images_path_list)
        dataset = chainer.datasets.TransformDataset(dataset, self.transform)
        return dataset

    def load_images_as_input(self):
        images_path_list = glob.glob('{}/*.jpg'.format(self.path))
        images_list = [Image.open(image_path)
                       for image_path in images_path_list]
        images_list = [np.asarray(image).transpose(2, 0, 1)
                       for image in images_list]
        images_array_list = map(self.transform, images_list)
        return np.asarray([s for s in images_array_list])

    def transform(self, img):
        img = img.astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize(self.size, Image.BICUBIC)
        img = np.asarray(img).transpose(2, 0, 1)
        img = img.astype(np.float32)
        img = img[0, :, :]
        img = img / 255
        img = img.reshape(-1)
        return img

    @staticmethod
    def save_image(data, savename, output_path, device=-1):
        destination = os.path.join(output_path, savename)
        num = 1
        if not os.path.exists(destination):
            os.mkdir(destination)

        if device >= 0:
            data = cuda.cupy.asnumpy(data)

        for image in data:
            im = image.reshape(height, width)
            im = im * 255
            pil_img = Image.fromarray(np.uint8(im)).convert('RGB')
            pil_img.save(os.path.join(destination, str(int(num+1))+'.png'))
            num += num


def train_autoencoder():
    datetime_str = datetime.now().strftime("%Y%m%d%H%M")
    NAME_OUTPUTDIRECTORY = 'exp' + datetime_str
    FILENAME_MODEL = 'ae_' + datetime_str + '.model'
    FILENAME_RESULT = 'result.txt'
    output_path = os.path.join('./result', NAME_OUTPUTDIRECTORY)
    os.mkdir(output_path)

    model = Autoencoder(width * height, hidden)
    target = ResizedImageDataset('./train_data', (width, height))
    train = target.load_images_as_dataset()
    train_iter = chainer.iterators.SerialIterator(train, batchsize)

    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model.to_gpu()

    opt = chainer.optimizers.Adam()
    opt.setup(model)
    loss_list = []
    train_num = len(train)

    for epoch in range(epoch_num):
        for i in range(0, train_num, batchsize):
            batch = train_iter.next()
            if gpu_id >= 0:
                x = cuda.cupy.asarray([s for s in batch])
            else:
                x = np.asarray([s for s in batch])
            loss = model(x)
            model.cleargrads()
            loss.backward()
            opt.update()
        loss_list.append(loss.array)
        print('epoch'+str(int(epoch+1))+':'+str(loss.data))
        # ロス値のグラフを出力
        plt.grid()
        plt.tight_layout()
        plt.plot([i for i, x in enumerate(loss_list, 1)], loss_list)
        plt.savefig(os.path.join(output_path, 'loss'))
    # 入力・出力画像のサンプルを保存
    y = model.forward(x)
    ResizedImageDataset.save_image(x, 'input', output_path, gpu_id)
    ResizedImageDataset.save_image(y.array, 'reconst', output_path, gpu_id)
    # モデルを保存
    model.to_cpu()
    serializers.save_npz(os.path.join(output_path, FILENAME_MODEL), model)
    # 条件をテキストファイルに出力
    with open(os.path.join(output_path, FILENAME_RESULT), mode='w') as f:
        f.write('width:'+str(width)+'\n')
        f.write('height:'+str(height)+'\n')
        f.write('hidden:'+str(hidden)+'\n')
        f.write('compression-rate:' +
                str(round((hidden/(width*height))*100, 2))+'%\n')


def test_autoencoder():
    target = ResizedImageDataset('./test', (width, height))
    test = target.load_images_as_input()
    model = Autoencoder(width*height, hidden)
    serializers.load_npz('./test/ae_201812250527.model', model)
    x = np.asarray(test)
    y = model.forward(x)
    ResizedImageDataset.save_image(x, 'input', './test/', -1)
    ResizedImageDataset.save_image(y.array, 'output', './test/', -1)


def main():
    # test_autoencoder()
    train_autoencoder()


if __name__ == '__main__':
    main()
