from skimage import io
import numpy as np
import matplotlib.pyplot as plt

''' define path and name '''
path = '/Users/yckao85/TKU/multimedia/05_07/'
src_name = 'src.jpg'
dst_name = 'dst.jpg'

''' read file and generate black result '''
src_img = io.imread(path + src_name)
dst_img = np.zeros(src_img.shape)

colors = ('red', 'green', 'blue')
def showHistogram(img):
  fig, ax = plt.subplots(3)
  for i in range(0, len(colors)):
    hist, edge = np.histogram(img[:,:,i], bins='auto')
    print(hist, edge)
    ax[i].bar(edge[:-1], hist, color=colors[i])
  plt.show()

def createMask(img, color_range):
  mask = (np.all(img > color_range[0], axis=2) & np.all(img < color_range[1], axis=2))
  return mask.reshape(mask.shape + (1, )).repeat(3, axis=2)

def applyMask(img, mask):
  return np.ma.array(img, mask=~mask).filled(255)

def mergeMask(masks):
  merged_mask = np.full(masks[0].shape, False)
  for i in range(0, len(masks)):
    merged_mask |= masks[i]
  return merged_mask

rgbRanges = [
  [[100, 0, 0], [255, 100, 100]],
  [[0, 0, 0], [50, 50, 50]],
  [[30, 0, 0], [255, 50, 50]],
  [[0, 0, 0], [100, 60, 100]]
]

masks = [createMask(src_img, rgbRanges[i]) for i in range(0, len(rgbRanges))]
merged_mask = mergeMask(masks)
  
dst_img = applyMask(src_img, merged_mask)

io.imsave(path+dst_name, dst_img.astype(np.uint8))
