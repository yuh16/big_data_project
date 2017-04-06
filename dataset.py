import os
import glob
import numpy as np
from scipy.misc import imread, imresize

class Dataset:
    def __init__(self, train_list,train_label):
        # Load training images (path) and labels

        # Init params
        self.train_image = train_list
        self.train_label = train_label
        self.test_image = train_list
        self.test_label = train_label
        
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_label)
        print self.train_size
        self.test_size = len(self.test_label)
        self.crop_size = 224
        self.scale_size = 224
        self.mean = np.array([104., 117., 124.])
        self.n_classes = 21
        #img, onelabel = self.next_batch( 10, 'test')
        
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None
        # Read images
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            img = imread(paths[i])
            h, w, c = img.shape
            assert c==3
            
            img = imresize(img, (self.scale_size, self.scale_size))
            img = img.astype(np.float32)
            img -= self.mean
            shift = int((self.scale_size-self.crop_size)/2)
            img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            images[i] = img_crop

        # Expand labels
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in xrange(len(labels)):
            one_hot_labels[i][labels[i]-1] = 1
        return images, one_hot_labels

if __name__ == '__main__':
    
    trainpaths = []
    paths = []
    t_labels = list()
    t_name = list()
    testpaths = []
    test_label = list()
    train_label = list()
    dir1 = os.path.dirname(__file__)
    rpath = "./new_data"
    i = 0
    j = 0
    filename = os.path.join(dir1, rpath)
    for root, dirs, files in os.walk(filename, topdown=False):
        for name in dirs:
            subdir = os.path.join(filename,name)
            j = j +1
            breed = j
            for pic in os.listdir(subdir):
                if pic.endswith(".jpg"):
                    paths.append(os.path.join(subdir,pic))
                    t_labels.append(breed)
                    t_name.append(name)
                    i = i+1
                    if i % 10 == 0:
                        testpaths.append(os.path.join(subdir,pic))
                        test_label.append(breed)
                    else:
                        trainpaths.append(os.path.join(subdir,pic))
                        train_label.append(breed)
                    
    train_ptr = 0  
    length = len(t_labels)      
    batch_size = 100
    dataset = Dataset(paths,t_labels)
    for k in range(0,length,batch_size):
        batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
        print tpr
        
        