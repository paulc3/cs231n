import random
import os
import numpy as np
from skimage import io
from skimage.filters import gaussian
from PIL import Image

MAIN_DIR = './raw/'
LABELS_SUBDIRS = ['amusement','awe','contentment','excitement',\
	'anger','disgust','fear','sadness']
transformations = ['BLURRED', 'FLIPPED']

for label_subdir in LABELS_SUBDIRS:
	if not (label_subdir == 'fear'): continue
	dir_path = MAIN_DIR + label_subdir + "/"
	for image_name in os.listdir(dir_path):
		if image_name.startswith('BLURRED') or image_name.startswith('FLIPPED'): continue
		print(image_name)
		original = io.imread(dir_path + image_name)
		transformation = random.choice(transformations)
		if transformation == 'BLURRED':
			new_image = gaussian(original, sigma=5, multichannel=True)
		else:
			new_image = np.fliplr(original)

		img = Image.fromarray(new_image, 'RGB')
		new_filepath = dir_path + transformation + "_" + image_name
		img.save(new_filepath)



