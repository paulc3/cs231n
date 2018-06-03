import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = np.genfromtxt('confusion_matrix_2_classes')
normalized_confusion_matrix = np.genfromtxt('normalized_confusion_matrix_2_classes')
normalized_supplementary_matrix = np.genfromtxt('normalized_supplementary_matrix_2_classes')
supplementary_matrix = np.genfromtxt('supplementary_matrix_2_classes')

def plot_confusion_matrix(confusion_matrix, name):
  fig = plt.figure(figsize=(10, 10))
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(confusion_matrix, cmap=plt.cm.jet, 
                  interpolation='nearest')
  
  width, height = confusion_matrix.shape
  
  for x in xrange(width):
    for y in xrange(height):
      ax.annotate("%.2f" % confusion_matrix[x][y], xy=(y, x), 
                  horizontalalignment='center',
                  verticalalignment='center')

  classes = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
  cb = fig.colorbar(res)
  plt.yticks(range(len(classes)), classes)
  if height == 2:
    plt.xticks(range(height), ['negative', 'positive'], rotation=45)
  else:
    plt.xticks(range(height), classes, rotation=45)
    
  plt.savefig(name + '.png', format='png')

plot_confusion_matrix(confusion_matrix, 'confusion_matrix')
plot_confusion_matrix(normalized_confusion_matrix, 'normalized_confusion_matrix')
plot_confusion_matrix(supplementary_matrix, 'supplementary_matrix')
plot_confusion_matrix(normalized_supplementary_matrix,
        'normalized_supplementary_matrix')
