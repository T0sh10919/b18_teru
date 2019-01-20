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

from statistics import mean, median, variance, stdev


width, height = 128, 72
hidden = 100
epoch_num = 1
batchsize = 12
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
        images_path_list = glob.glob('{}/*.png'.format(self.path))
        dataset = chainer.datasets.ImageDataset(images_path_list)
        dataset = chainer.datasets.TransformDataset(dataset, self.transform)
        return dataset

    def load_images_as_input(self):
        images_path_list = glob.glob('{}/*.png'.format(self.path))
        out = []
        for item in images_path_list:
            with Image.open(item) as image:
                image = np.asarray(image).transpose(2, 0, 1)
                image = self.transform(image)
                out.append(image)
        return np.asarray([s for s in out])



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
    def save_image(data, savename, output_path, device=0):
        destination = os.path.join(output_path, savename) #output_path = ./result/exphoge, destination = ./result/exphoge/input or reconst etc
        num = 0
        if not os.path.exists(destination):
            os.mkdir(destination)

        if device >= 0:
            data = cuda.cupy.asnumpy(data)

        for image in data:
            im = image.reshape(height, width)
            im = im * 255
            pil_img = Image.fromarray(np.uint8(im)).convert('RGB')
            pil_img.save(os.path.join(destination, '{0:06d}'.format(num) +'.png'))
            
            num += 1


def train_autoencoder():
    datetime_str = datetime.now().strftime("%Y%m%d%H%M")
    NAME_OUTPUTDIRECTORY = 'exp' + datetime_str
    FILENAME_MODEL = 'ae_' + datetime_str + '.model'
    FILENAME_RESULT = 'result.txt'
    output_path = os.path.join('./result', NAME_OUTPUTDIRECTORY)
    os.mkdir(output_path)

    model = Autoencoder(width * height, hidden)
    target = ResizedImageDataset('./train_sample', (width, height)) #パス指定change
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

    #trainの再構成誤差算出
    target_train = ResizedImageDataset('./train_sample', (width, height)) #パス指定change
    all_train = target_train.load_images_as_input()
    x_train = np.asarray(all_train)
    y_train = model.forward(x_train)
    hist_list = []
    reconst_error(x_train, y_train, hist_list)
    #閾値算出
    m = mean(hist_list)
    print('平均は' + str(m))
    std = stdev(hist_list)
    sikiti = m + 3 * std
    print('閾値は' + str(sikiti))
    #ヒストグラム作成
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(hist_list, bins=128)
    ax.set_xlabel('error')
    ax.set_ylabel('sample') 
    ax.axvline(x=sikiti, color='red')
    plt.savefig(os.path.join(output_path, 'histgram_train'))
    plt.show()


    #正常test_dataの再構成誤差算出
    target_test = ResizedImageDataset('./test_normal', (width, height)) #パス指定change
    test = target_test.load_images_as_input()
    x_test = np.asarray(test)
    y_test = model.forward(x_test)
    ResizedImageDataset.save_image(x_test, 'input_test',
                                   output_path, 0)
    ResizedImageDataset.save_image(y_test.array, 'reconst_test',
                                   output_path, 0)
    error_list1 = []
    anomalous_list1 = []
    reconst_error(x_test, y_test, error_list1)

    for error in error_list1:
        if error >= sikiti:
            anomalous_list1.append(error_list1.index(error)) 
        else:
            pass
    
    print(anomalous_list1)
    ratio1 = ((len(error_list1)-len(anomalous_list1))/len(error_list1))*100 #(全体-異常)/全体
    print('検知した数/全体(%) = ' + str(ratio1))
    #棒グラフ作成
    bar_graph(error_list1, sikiti, output_path, 'bar_test_normal')

    #異常test_dataの再構成誤差算出
    target_test = ResizedImageDataset('./test_anomalous', (width, height)) #パス指定change
    test = target_test.load_images_as_input()
    x_test = np.asarray(test)
    y_test = model.forward(x_test)
    ResizedImageDataset.save_image(x_test, 'input_test',
                                   output_path, 0)
    ResizedImageDataset.save_image(y_test.array, 'reconst_test',
                                   output_path, 0)
    error_list2 = []
    anomalous_list2 = []
    reconst_error(x_test, y_test, error_list2)
    for error in error_list2:
        if error >= sikiti:
            anomalous_list2.append(error_list2.index(error)) 
        else:
            pass
    print(anomalous_list2)
    ratio2 = (len(anomalous_list2)/len(error_list2))*100

    print('検知した数/全体(%) = ' + str(ratio2))
    #異常画像リスト作成
    os.mkdir(output_path + '/anomalous_img')
    path_to_test_data = './test_anomalous' #パス指定change
    image_path_list = sorted(glob.glob('{}/*.png'.format(path_to_test_data)))
    anomalous_img_path = []
    for index in anomalous_list2:
        print(index)
        b = image_path_list[index]
        anomalous_img_path.append(b)
        #print(b)
    
    anomalous_image_list = []
    for anomalous_image in anomalous_img_path:
        a = Image.open(anomalous_image)
        anomalous_image_list.append(a)
    #print(anomalous_image_list)
    num = 0
    for im in anomalous_image_list:
        im.save(output_path + '/anomalous_img/' + '{0:06d}'.format(num) +'.png')
        num += 1


    #棒グラフ作成 
    bar_graph(error_list2, sikiti, output_path, 'bar_test_anomalous')

    # 条件をテキストファイルに出力
    with open(os.path.join(output_path, FILENAME_RESULT), mode='w') as f:
        f.write('width:'+str(width)+'\n')
        f.write('height:'+str(height)+'\n')
        f.write('hidden:'+str(hidden)+'\n')
        f.write('compression-rate:' +
                str(round((hidden/(width*height))*100, 2))+'%\n')
        f.write('正常と検知した数/全体(%) = '+str(round((ratio1), 2))+'%\n')
        f.write('異常と検知した数/全体(%) = '+str(round((ratio2), 2))+'%\n')
        f.write('train_data:' + '\n')
        f.write('結果:' + '\n')
        f.write('考察:' + '\n')
        f.write('展望:' + '\n')

def test_autoencoder():
    target = ResizedImageDataset('./test_data(100)', (width, height))
    test = target.load_images_as_input()
    model = Autoencoder(width*height, hidden)
    serializers.load_npz('./test/ae_201901161316.model', model)
    x = np.asarray(test)
    y = model.forward(x)


    ResizedImageDataset.save_image(x, 'input', './test/', 0) #save_image(data, savename, output_path, device=0)
    ResizedImageDataset.save_image(y.array, 'output', './test/', 0)

#棒グラフ作成    
def bar_graph(error_list_name, thresh, output_path_name, bar_savename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    left = np.array(list(range(0, len(error_list_name))))
    ax.bar(left, np.array(error_list_name))
    ax.set_xlabel('sample')
    ax.set_ylabel('reconst_error') 
    ax.axhline(y=thresh, color='red')
    plt.savefig(os.path.join(output_path_name, bar_savename))
    plt.show()


def reconst_error(x_data, y_data, error_list_name):
    for j in range(len(x_data)):
        diff = abs(x_data[j] - y_data.array[j])
        s = (np.sum(np.power(diff, 2)))/len(diff) 
        #差分したもののそれぞれの要素を二乗して要素を足し合わせたもの/(要素数)
        error_list_name.append(s)#error_listの中身は再構成誤差

    


def main():
    #test_autoencoder()
    train_autoencoder()


if __name__ == '__main__':
    main()