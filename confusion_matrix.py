import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

confusion_matrix = np.genfromtxt('confusion_matrix_8_classes')

def plot_confusion_matrix(confusion_matrix, name):
  fig = plt.figure(figsize=(10, 10))
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(confusion_matrix, cmap=plt.cm.jet, 
                  interpolation='nearest')
  
  width, height = confusion_matrix.shape
  
  for x in range(width):
    for y in range(height):
      ax.annotate("%.2f" % confusion_matrix[x][y], xy=(y, x), 
                  horizontalalignment='center',
                  verticalalignment='center', color='white', fontsize=14, fontweight='bold')

  if height == 2:
    classes = ['positive', 'negative']
    cb = fig.colorbar(res)
    plt.yticks(range(len(classes)), classes)
    plt.xticks(range(height), ['negative', 'positive'], rotation=45)
  else:
    classes = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    cb = fig.colorbar(res)
    plt.yticks(range(len(classes)), classes)
    plt.xticks(range(height), classes, rotation=45)
    
  plt.savefig(name + '.png', format='png')

plot_confusion_matrix(confusion_matrix, 'confusion_matrix')
