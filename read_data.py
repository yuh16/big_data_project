#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:20:54 2017

@author: yuhan
"""
from __future__ import division

import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.misc import imread, imresize
from itertools import groupby
from collections import defaultdict
from dataset import Dataset
#from input2records import *

def write_records_file(dataset, record_location):
    """
    Fill a TFRecords file with the images found in `dataset` and include their category.

    Parameters
    ----------
    dataset : dict(list)
      Dictionary with each key being a label for the list of image filenames of its value.
    record_location : str
      Location to store the TFRecord output.
    """
    writer = None
    #sess = tf.InteractiveSession()
    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 10 == 0:
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
                print writer
            current_index += 1
            image_file = tf.read_file(image_filename)

            # In ImageNet dogs, there are a few images which TensorFlow doesn't recognize as JPEGs. This
            # try/catch will ignore those images.
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # Converting to grayscale saves processing and memory but isn't required.
            #grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(image, (224,224))

            # tf.cast is used here because the resized images are floats but haven't been converted into
            # image floats where an RGB value is between [0,1).
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            # Instead of using the label as a string, it'd be more efficient to turn it into either an
            # integer index or a one-hot encoded rank one tensor.
            # https://en.wikipedia.org/wiki/One-hot
            image_label = breed.encode("utf-8")

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())
    writer.close()



#write_records_file(testing_dataset, "./data/new_data/test_records/tfrecods")
#write_records_file(training_dataset, "./data/new_data/training_records/tfrecods")

def read_and_decode(filename):
    #根据文件名生成一个队列
    sess = tf.InteractiveSession()
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [224, 224, 1]) #3
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label





if __name__ == '__main__':
    
    sess = tf.InteractiveSession()
    filepaths = []
    directory_list = list()
    dir = os.path.dirname(__file__)
    rpath = "./new_data"
    filename = os.path.join(dir, rpath)
    for root, dirs, files in os.walk(filename, topdown=False):
        for name in dirs:
            subdir = os.path.join(filename,name)
            for pic in os.listdir(subdir):
                if pic.endswith(".jpg"):
                    filepaths.append(os.path.join(subdir,pic))
                    directory_list.append(name)
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    image_filenames = glob.glob("/new_data/*/*.jpg")
    image_filename_with_breed = map(lambda filename: (filename.split("/")[2], filename), image_filenames)
    for breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        # Enumerate each breed's image and send ~20% of the images to a testing set
        for i, breed_image in enumerate(breed_images):
            #print breed_image
            if i % 5 == 0:
                testing_dataset[breed].append(breed_image[1])
            else:
                training_dataset[breed].append(breed_image[1])
            
        breed_training_count = len(training_dataset[breed])
        breed_testing_count = len(testing_dataset[breed])     
        assert round(breed_testing_count / (breed_training_count + breed_testing_count), 2) > 0.18, "Not enough testing images."
        
    dataset = Dataset(filepaths)


       
    #img, label = read_and_decode(tf.train.match_filenames_once("./data/new_data/tfrecords/*.tfrecords"))
    #img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                            batch_size=10, capacity=200,
    #                                            min_after_dequeue=100)
