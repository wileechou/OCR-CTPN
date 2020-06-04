#coding:utf-8
import img_backward
import tensorflow as tf
from img_backward import *
import time
TEST_INTERVAL_SECS = 5

def test(features,labels):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, img_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, img_forward.OUTPUT_NODE])
        y = img_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(img_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
		
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(img_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x:features, y_: labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    features,labels=img_forward.load_ocr_data('chinese2/')
    test(features,labels)

if __name__ == '__main__':
    main()
