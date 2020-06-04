import tensorflow as tf
import os
from pylab import *
from PIL import Image
INPUT_NODE = 272
OUTPUT_NODE = 3495
LAYER1_NODE = 500

def load_ocr_data(path):
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
    features = []
    labels = []


    for imname in imlist:
        label = int((imname.split('/')[1]).split('_')[0])
        label1=[0.]*3495
        label1[label%3495]=1.
        labels.append(label1)

        im = array(Image.open(imname).convert('L'))
        im = im.astype(np.float32)
        im = im = (1. / 255) * im + 0
        im = [i for item in im for i in item]
        features.append(im)

    return features, labels


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b
	
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
