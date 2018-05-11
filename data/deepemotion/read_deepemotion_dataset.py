# This file is used to read in and process the deepemotion dataset.
# See http://www.cs.rochester.edu/u/qyou/deepemotion/.

import csv
from collections import Counter
from skimage import io
import urllib
import tensorflow as tf
from PIL import Image

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

disk_filepath = "/Volumes/Frank's Hard Drive/Stanford 2017-2018/cs231n/data/deepemotion"

def has_sufficient_upvotes(downvotes, upvotes):
	return upvotes + downvotes > 0 and upvotes / (upvotes + downvotes) > 0.75

def download_image(url):
	image = io.imread(url)
	if len(image.shape) != 3 or image.shape[2] != 3:
		raise ValueError('invalid image dims: ', image.shape) # bad dims, raise err to be caught by caller
	return image

def write_to_tfrecord_file(image, label):
	def _int64_feature(value):
  		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def _bytes_feature(value):
  		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	encoded_image_str = tf.image.encode_jpeg(image).eval(session=tf.Session())
	H, W, C = image.shape
	example = tf.train.Example(features=tf.train.Features(feature={
	    # 'height': _int64_feature(H),
	    # 'width': _int64_feature(W),
	    # 'depth': _int64_feature(C),
	    'label': _int64_feature(int(label)),
	    'image_raw': _bytes_feature(encoded_image_str)}))
	writer.write(example.SerializeToString())

insufficent_upvotes = 0
sufficient_upvotes = 0
good_urls = 0
bad_urls = 0
instances_of_label = Counter()
total_height = 0
total_width = 0
writer = tf.python_io.TFRecordWriter('data.tfrecords')

with open('sorted_raw_data.csv', 'r') as file:
	reader = csv.reader(file)
	for emotion, url, downvotes, upvotes in reader:
		if not has_sufficient_upvotes(float(downvotes), float(upvotes)):
			insufficent_upvotes += 1
			continue
		sufficient_upvotes += 1
		instances_of_label[emotion] += 1

		label = EMOTION_TO_LABEL[emotion]

		# Fetch image from URL. If throws exception, count it and skip it.
		try:
			# creates a numpy array shaped [H, W, C]
			image = download_image(url)
			good_urls += 1
			total_height += image.shape[0]
			total_width += image.shape[1]
		except (urllib.error.HTTPError, ValueError):
			bad_urls += 1
			continue

		# for now just resize to this standard size by cropping/padding
		# we have a couple options to explore here later, like resizing 
		# using interpolation instead of simply cropping/padding. can also
		# explore data augmentation like filters, flipping, etc.
		image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

		# convert this example to TFRecord and write to data.tfrecord file
		# write_to_tfrecord_file(image, label)

		# write image to disk
		image_path = '{}/{}/{}-{}.jpg'.format(disk_filepath, emotion, emotion, instances_of_label[emotion])
		img = Image.fromarray(image.eval(session=tf.Session()), 'RGB')	# leverage PIL.Image lib
		img.save(image_path)

		if good_urls > 1000: 
			break

writer.close()
print("insufficent_upvotes: " + str(insufficent_upvotes))
print("sufficient_upvotes: " + str(sufficient_upvotes))
print('good urls: ' + str(good_urls))
print('bad urls: ' + str(bad_urls))
print('mean height: ' + str(total_height/good_urls))
print('mean width: ' + str(total_width/good_urls))
print(instances_of_label)