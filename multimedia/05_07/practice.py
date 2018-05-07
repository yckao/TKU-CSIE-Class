'''This Module is classroom practice in 05_07'''

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

''' define path and name '''
PATH = '/Users/yckao85/TKU/multimedia/05_07/'
SRC_NAME = 'src.jpg'
DST_NAME = 'dst.jpg'

''' read file and generate black result '''
src_img = io.imread(PATH + SRC_NAME)
dst_img = np.zeros(src_img.shape)

''' define colors in rgb format '''
colors = ('red', 'green', 'blue')
''' show histogram using matplotlib '''
def show_histogram(img):
    _, ax = plt.subplots(3)
    for i in range(0, len(colors)):
        hist, edge = np.histogram(img[:, :, i], bins='auto')
        ax[i].bar(edge[:-1], hist, color=colors[i])
    plt.show()

''' create mask with image and color range '''
def create_mask(img, color_range):
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
def apply_mask(img, mask):
    return np.ma.array(img, mask=~mask).filled(255)

''' merge masks to single mask '''
def merge_mask(masks):
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
masks = [create_mask(src_img, rgb_ranges[i]) for i in range(0, len(rgb_ranges))]

''' merge all masks to single mask'''
merged_mask = merge_mask(masks)

''' get dst_img by apply mask '''
dst_img = apply_mask(src_img, merged_mask)

''' save image and done! '''
io.imsave(PATH+DST_NAME, dst_img.astype(np.uint8))
