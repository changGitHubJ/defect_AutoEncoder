import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from PIL import Image

TRAIN_DATA_SIZE = 120
TEST_DATA_SIZE = 30
IMG_SIZE = 256
OUTPUT_SIZE = 256*256
NO_DEFECT = 0.2
DEFECT = 0.8

if __name__ == "__main__":

    init = tf.global_variables_initializer()
    sess = tf.Session()
    with sess.as_default():

        if not os.path.exists('./data'):
            os.mkdir('./data')

        # remove old file
        if(os.path.exists('./data/trainImage256.txt')):
           os.remove('./data/trainImage256.txt')
        if(os.path.exists('./data/testImage256.txt')):
           os.remove('./data/testImage256.txt')
        if(os.path.exists('./data/trainLABEL' + str(IMG_SIZE) + '.txt')):
            os.remove('./data/trainLABEL' + str(IMG_SIZE) + '.txt')
        if(os.path.exists('./data/trainWEIGHT' + str(IMG_SIZE) + '.txt')):
            os.remove('./data/trainWEIGHT' + str(IMG_SIZE) + '.txt')
        if(os.path.exists('./data/testLABEL' + str(IMG_SIZE) + '.txt')):
            os.remove('./data/testLABEL' + str(IMG_SIZE) + '.txt')
        if(os.path.exists('./data/testWEIGHT' + str(IMG_SIZE) + '.txt')):
            os.remove('./data/testWEIGHT' + str(IMG_SIZE) + '.txt')

        for k in range(TRAIN_DATA_SIZE + TEST_DATA_SIZE):
            filename = './data/Class1_def/' + str(k + 1) + '.png'
            print(filename)
            imgtf = tf.read_file(filename)
            img = tf.image.decode_png(imgtf, channels=1)
            resized = tf.image.resize_images(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
            array = resized.eval()
            line = str(k)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    line = line + ',' + str(array[i, j, 0])
            line = line + '\n'
            if(k < TRAIN_DATA_SIZE):
                file = open('./data/trainImage256.txt', 'a')
                file.write(line)
                file.close()
            else:
                file = open('./data/testImage256.txt', 'a')
                file.write(line)
                file.close()

        # label #
        trnLABEL = []
        trnWEIGHT = []
        tstLABEL = []
        tstWEIGHT = []    
        # defection data
        x = np.linspace(1.0, 511, IMG_SIZE)
        y = np.linspace(1.0, 511, IMG_SIZE)
        print('reading Class1_def')
        label1 = open('./data/Class1_def/labels.txt', 'r')
        for k in range(TRAIN_DATA_SIZE + TEST_DATA_SIZE):
            line = label1.readline()
            val = line.split('\t')
            num = int(val[0]) - 1
            mjr = float(val[1])
            mnr = float(val[2])
            rot = float(val[3])
            cnx = float(val[4])
            cny = float(val[5]) 

            # inverse rotate pixels
            label = np.zeros([OUTPUT_SIZE + 1])
            weight = np.zeros([OUTPUT_SIZE + 1])
            label[0] = num # index
            weight[0] = num # index
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    dist = math.sqrt((x[i] - cnx)**2 + (y[j] - cny)**2)
                    xTmp = (x[i] - cnx) * math.cos(-rot) - (y[j] - cny) * math.sin(-rot)
                    yTmp = (x[i] - cnx) * math.sin(-rot) + (y[j] - cny) * math.cos(-rot)
                    ang = math.atan(yTmp/xTmp)
                    distToEllipse = math.sqrt((mjr * math.cos(ang))**2 + (mnr * math.sin(ang))**2)
                    if(dist < distToEllipse):
                        label[j*IMG_SIZE + i + 1] = 1 # defection
                        weight[j*IMG_SIZE + i + 1] = DEFECT
                    else:
                        label[j*IMG_SIZE + i + 1] = 0
                        weight[j*IMG_SIZE + i + 1] = NO_DEFECT
            
            # plot test
            #if(k == 0):
                #plt.figure(figsize=(5, 5))
                #z = label[1:OUTPUT_IMG_SIZE*OUTPUT_IMG_SIZE + 1].reshape([OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE])
                #plt.imshow(z)
                #plt.show()
        
            if(k < TRAIN_DATA_SIZE):
                trnLABEL.append(label)
                trnWEIGHT.append(weight)
            else:
                tstLABEL.append(label)
                tstWEIGHT.append(weight)

        # normalize
        w_array = np.array(trnWEIGHT)
        for k in range(TRAIN_DATA_SIZE):
            s = sum(w_array[k, 1:OUTPUT_SIZE + 1])
            w_array[k, 1:OUTPUT_SIZE + 1] = w_array[k, 1:OUTPUT_SIZE + 1]/s
        trnWEIGHT = w_array.tolist()
        
        w_tst_array = np.array(tstWEIGHT)
        for k in range(TEST_DATA_SIZE):
            s = sum(w_tst_array[k, 1:OUTPUT_SIZE + 1])
            w_tst_array[k, 1:OUTPUT_SIZE + 1] = w_tst_array[k, 1:OUTPUT_SIZE + 1]/s
        tstWEIGHT = w_tst_array.tolist()
        
        np.savetxt('./data/trainLABEL' + str(IMG_SIZE) + '.txt', trnLABEL, fmt='%d', delimiter=',')
        np.savetxt('./data/trainWEIGHT' + str(IMG_SIZE) + '.txt', trnWEIGHT, fmt='%.5f', delimiter=',')
        np.savetxt('./data/testLABEL' + str(IMG_SIZE) + '.txt', tstLABEL, fmt='%d', delimiter=',')
        np.savetxt('./data/testWEIGHT' + str(IMG_SIZE) + '.txt', tstWEIGHT, fmt='%.5f', delimiter=',')

        sess.close()
