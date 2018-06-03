import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = np.genfromtxt('confusion_matrix_8_classes')
normalized_confusion_matrix = np.genfromtxt('normalized_confusion_matrix_8_classes')

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
  plt.yticks(range(width), classes)
  plt.xticks(range(height), classes, rotation=45)
    
  plt.savefig(name + '.png', format='png')

plot_confusion_matrix(confusion_matrix, 'confusion_matrix')
plot_confusion_matrix(normalized_confusion_matrix, 'normalized_confusion_matrix')
