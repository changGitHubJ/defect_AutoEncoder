import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

NUM = 50
IMG_SIZE = 256
OUTPUT_SIZE = 256*256

def readImages(filename):
    images = np.zeros((NUM, IMG_SIZE*IMG_SIZE))
    fileImg = open(filename)
    for k in range(NUM):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(IMG_SIZE*IMG_SIZE):
            images[k, i] = float(val[i + 1])
    return images

def readLabels(filename):
    labels = np.zeros((NUM, OUTPUT_SIZE))
    fileImg = open(filename)
    for k in range(NUM):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(OUTPUT_SIZE):
            labels[k, i] = float(val[i + 1])
    return labels

if __name__=='__main__':
    tst_image = readImages('./data/testImage256.txt')
    tst_label = readLabels('./data/testWEIGHT256.txt')

    for i in range(NUM):
        plt.figure(figsize=[10, 3])
        plt.subplot(1, 3, 1)
        fig = plt.imshow(tst_image[i, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        plt.subplot(1, 3, 2)
        fig = plt.imshow(tst_label[i, :].reshape([IMG_SIZE, IMG_SIZE]), cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        rslt = Image.open('./result/' + str(i) + '.png')
        plt.subplot(1, 3, 3)
        fig = plt.imshow(rslt)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.show()
