import tensorflow as tf
import os
from pylab import *
from PIL import Image
import img_backward
import img_forward

def load_ocr_data1(path):
    features=[]
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
    for imname in imlist:
        reim = (Image.open(imname)).resize((16, 17),Image.ANTIALIAS)
        im = array(reim.convert('L'))
        im = im.astype(np.float32)
        im = im = (1. / 255) * im + 0
        im = [i for item in im for i in item]
        features.append(im)
    return features
def restore_model(features):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, img_forward.INPUT_NODE])
        y = img_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(img_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(img_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: features})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


