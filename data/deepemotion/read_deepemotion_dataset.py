# This file is used to read in and process the deepemotion dataset.
# See http://www.cs.rochester.edu/u/qyou/deepemotion/.

import csv
from collections import Counter
from skimage import io
import urllib
import tensorflow as tf
from PIL import Image
import os

# Positive emotions first, then negative emotions. This dict shouldn't be
# modified.
EMOTION_TO_LABEL = {
        'amusement': 0,
        'awe': 1,
        'contentment': 2,
        'excitement': 3,
        'anger': 4,
        'disgust': 5,
        'fear': 6,
        'sadness': 7
}

disk_filepath = "./raw"

def has_sufficient_upvotes(downvotes, upvotes):
        return upvotes + downvotes > 0 and upvotes / (upvotes + downvotes) > 0.75

def download_image(url):
        image = io.imread(url)
        if len(image.shape) != 3 or image.shape[2] != 3:
               # bad dims, raise err to be caught by caller
               raise ValueError('invalid image dims: ', image.shape)
        return image

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Write examples to tfrecord file.
def write_to_tfrecord_file(input_image, label):
    g = tf.Graph()
    with g.as_default():
        uncropped_image = tf.placeholder(tf.uint8)
        
        # for now just resize to this standard size by cropping/padding
        # we have a couple options to explore here later, like resizing 
        # using interpolation instead of simply cropping/padding. can also
        # explore data augmentation like filters, flipping, etc.
        cropped_image = tf.image.resize_image_with_crop_or_pad(
            uncropped_image, 224, 224)
        cropped_jpeg = tf.image.encode_jpeg(cropped_image)
        
        init = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        jpeg = sess.run(cropped_jpeg,
                        feed_dict={ uncropped_image: input_image })

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': int64_feature(int(label)),
        'image_raw': bytes_feature(jpeg)}))
    writer.write(example.SerializeToString())
    return image

insufficent_upvotes = 0
sufficient_upvotes = 0
good_urls = 0
bad_urls = 0
instances_of_label = Counter()
total_height = 0
total_width = 0
writer = None
prev_emotion = None

with open('sorted_raw_data.csv', 'r') as file:
        reader = csv.reader(file)
        for emotion, url, downvotes, upvotes in reader:
               if not has_sufficient_upvotes(float(downvotes), float(upvotes)):
                     insufficent_upvotes += 1
                     continue
               if emotion == "awe" or emotion == "anger" or emotion == "amusement":
                     continue
               sufficient_upvotes += 1
               instances_of_label[emotion] += 1

               label = EMOTION_TO_LABEL[emotion]

               # Fetch image from URL. If throws exception, count it and skip it.
               try:
                     # creates a numpy array shaped [H, W, C]
                     # The datatype is uint8.
                     image = download_image(url)
                     
                     good_urls += 1
                     total_height += image.shape[0]
                     total_width += image.shape[1]
               except (urllib.error.HTTPError, ValueError):
                     bad_urls += 1
                     continue

               # convert this example to TFRecord and write to data.tfrecord file
               if emotion != prev_emotion:
                     writer = tf.python_io.TFRecordWriter(
                         './tfrecords/{}.tfrecords'.format(emotion))
                     prev_emotion = emotion
                    
                     images_directory = '{}/{}'.format(disk_filepath, emotion)
                     if not os.path.exists(images_directory):
                         os.mkdir(images_directory)

               # Write image to disk.
               image_path = '{}/{}/{}-{}.jpg'.format(
                   disk_filepath, emotion, emotion, instances_of_label[emotion])
               img = Image.fromarray(image, 'RGB')
               # leverage PIL.Image lib
               img.save(image_path)

               write_to_tfrecord_file(image, label)

writer.close()
print("insufficent_upvotes: " + str(insufficent_upvotes))
print("sufficient_upvotes: " + str(sufficient_upvotes))
print('good urls: ' + str(good_urls))
print('bad urls: ' + str(bad_urls))
print('mean height: ' + str(total_height/good_urls))
print('mean width: ' + str(total_width/good_urls))
print(instances_of_label)