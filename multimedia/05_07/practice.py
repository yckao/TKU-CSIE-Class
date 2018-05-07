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

''' show histogram using matplotlib '''
colors = ('red', 'green', 'blue')
def showHistogram(img):
  fig, ax = plt.subplots(3)
  for i in range(0, len(colors)):
    hist, edge = np.histogram(img[:,:,i], bins='auto')
    print(hist, edge)
    ax[i].bar(edge[:-1], hist, color=colors[i])
  plt.show()

''' create mask with image and color range '''
def createMask(img, color_range):
  ''' 
    notice np array compare operator behavior [1, 2, 2] > [1, 1, 1] => [False, True, True]
    so we need to use np.all reduce mask to False
  '''
  mask = (np.all(img > color_range[0], axis=2) & np.all(img < color_range[1], axis=2))
  '''
    since we want to use numpy.masked_array, mask must be same shape with data
    so we need to repeat 3 times in axis=2
    before doing that, reshape mask from (y, x,) to (y, x, 1,)
  '''
  return mask.reshape(mask.shape + (1, )).repeat(3, axis=2)

''' apply mask to image by creating numpy.masked_array and filled with 255 '''
def applyMask(img, mask):
  return np.ma.array(img, mask=~mask).filled(255)

''' merge masks to single mask '''
def mergeMask(masks):
  ''' numpy.fill will fill shape with value (np.zeros(shape) equiv np.full(shape, 0)) '''
  merged_mask = np.full(masks[0].shape, False)
  for i in range(0, len(masks)):
    merged_mask |= masks[i]
  return merged_mask

''' define rgb ranges (the value generate manually, and only for the sample image ) '''
rgb_ranges = [
  [[100, 0, 0], [255, 100, 100]],
  [[0, 0, 0], [50, 50, 50]],
  [[30, 0, 0], [255, 50, 50]],
  [[0, 0, 0], [100, 60, 100]]
]

''' show histrogram of rgb color in src_img (uncomment if needed) '''
# showHistogram(src_img)
''' create mask by using generator with rgb_ranges '''
masks = [createMask(src_img, rgb_ranges[i]) for i in range(0, len(rgb_ranges))]
''' merge all masks to single mask'''
merged_mask = mergeMask(masks)

''' get dst_img by apply mask '''
dst_img = applyMask(src_img, merged_mask)

''' save image and done! '''
io.imsave(path+dst_name, dst_img.astype(np.uint8))
