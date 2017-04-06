import tensorflow as tf
import sys
import vgg16_orig
from dataset import Dataset
from vgg16_orig import vgg16
from datetime import datetime


def get_path_label(pstr):
    trainpaths = []
    paths = []
    labels = list()
    testpaths = []
    test_label = list()
    train_label = list()
    dir1 = os.path.dirname(__file__)
    rpath = pstr
    i = 0
    j = 1
    filename = os.path.join(dir1, rpath)
    for root, dirs, files in os.walk(filename, topdown=False):
        for name in dirs:
            subdir = os.path.join(filename,name)
            breed = j
            for pic in os.listdir(subdir):
                if pic.endswith(".jpg"):
                    paths.append(os.path.join(subdir,pic))
                    labels.append(breed)
                    if i % 10 == 0:
                        testpaths.append(os.path.join(subdir,pic))
                        test_label.append(breed)
                    else:
                        trainpaths.append(os.path.join(subdir,pic))
                        train_label.append(breed)
                    i = i+1
            j = j +1
    return trainpaths, train_label, testpaths, test_label

def main():
    # Learning params
    learning_rate = 0.001
    training_iters = 12800 # 10 epochs
    batch_size = 50
    display_step = 20
    test_step = 640 # 0.5 epoch

    # Network params
    n_classes = 20
    keep_rate = 0.5

    # Graph input
    # Launch the graph
    with tf.Session() as sess:
        batch_size = 20
        print 'Init variable'
        sess.run(init)
        # Load pretrained model
        vgg = vgg16_orig.vgg16(imgs, 'vgg16_weights.npz', sess)
        
        
        x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_var = tf.placeholder(tf.float32)
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        # Model
        
        #pred = vgg16.alexnet(x, keep_var)
        pred = vgg.probs
        # Loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
        # Evaluation
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        # Init
        init = tf.initialize_all_variables()
        trainpaths, train_label, testpaths, test_label = get_path_label("./new_data")
        # Load dataset
        dataset = Dataset(trainpaths,train_label)
        
        print 'Start training'
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
           
            # Display testing status
            if step%test_step == 0:
                test_acc = 0.
                test_count = 0
                for _ in range(int(dataset.test_size/batch_size)):
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print >> sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc)


            # Display training status
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc)
     
            step += 1
        print "Finish!"

if __name__ == '__main__':
    main()

