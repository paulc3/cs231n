import tensorflow as tf

def decode_tfrecords(example):
	print(example)
	feature_list = {'image_raw': tf.FixedLenFeature([], tf.string),
					'label': tf.FixedLenFeature([], tf.int64)}
	parsed_features = tf.parse_single_example(example, feature_list)
	image = tf.image.decode_jpeg(parsed_features['image_raw'])
	label = tf.cast(parsed_features['label'], tf.int32)
	return [image, label]

dataset = tf.data.TFRecordDataset('./data/deepemotion/data.tfrecords')
dataset = dataset.map(decode_tfrecords)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
	sess.run(iterator.initializer)
	images, labels = sess.run(next_element) # shapes [N, H, W, C] and [N,]