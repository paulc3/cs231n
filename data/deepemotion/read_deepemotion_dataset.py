# This file is used to read in and process the deepemotion dataset.
# See http://www.cs.rochester.edu/u/qyou/deepemotion/.

import csv
from collections import Counter

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

def has_sufficient_upvotes(downvotes, upvotes):
	return upvotes + downvotes > 0 and upvotes / (upvotes + downvotes) > 0.75


insufficent_upvotes = 0
sufficient_upvotes = 0

instances_of_label = Counter()

with open('sorted_raw_data.csv', 'rb') as file:
	reader = csv.reader(file)
	for emotion, url, downvotes, upvotes in reader:

		if not has_sufficient_upvotes(float(downvotes), float(upvotes)):
			insufficent_upvotes += 1
			continue
		sufficient_upvotes += 1
		instances_of_label[emotion] += 1

		label = EMOTION_TO_LABEL[emotion]

		# Do something with label...

		# Fetch image from URL. If throws exception, count it and skip it.

		# Process it into a tensor using
		# https://www.tensorflow.org/api_docs/python/tf/image/decode_image.
		# For cropping/resizing:
		# https://www.tensorflow.org/api_guides/python/image#Encoding_and_Decoding



print("insufficent_upvotes: " + str(insufficent_upvotes))
print("sufficient_upvotes: " + str(sufficient_upvotes))
print(instances_of_label)