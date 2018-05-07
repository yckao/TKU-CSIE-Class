from numpy import linalg
from skimage import io
import numpy as np

''' split rect area by given points
	ex. [0,0], [1, 2], [5, 5]
	[[0, 0], [1, 0], [1, 2]],
	[[1, 0], [5, 0], [5, 2]],
	[[0, 2], [1, 2], [1, 5]],
	[[1, 2], [5, 2], [5, 5]]
	(4 area)
'''
def split_area_by_point(points):
	sorted_x = list(map(lambda p : p[0],sorted(points, key = lambda p : p[0])))
	sorted_y = list(map(lambda p : p[1],sorted(points, key = lambda p : p[1])))
	areas = []
	for y in range(0, len(sorted_y) - 1):
		for x in range(0, len(sorted_x) - 1):
			areas.append([[sorted_x[x], sorted_y[y]], [sorted_x[x+1], sorted_y[y]], [sorted_x[x+1], sorted_y[y+1]]])
	return areas

'''
solve a, b in y = ax + b
'''
def solve(src_points, dst_points):
	x = [
		[src_points[0][0], src_points[0][1], 0, 0, 1, 0],
		[0, 0, src_points[0][0], src_points[0][1], 0, 1],
		[src_points[1][0], src_points[1][1], 0, 0, 1, 0],
		[0, 0, src_points[1][0], src_points[1][1], 0, 1],
		[src_points[2][0], src_points[2][1], 0, 0, 1, 0],
		[0, 0, src_points[2][0], src_points[2][1], 0, 1],
	]
	
	y = [
		[dst_points[0][0]],
		[dst_points[0][1]],
		[dst_points[1][0]],
		[dst_points[1][1]],
		[dst_points[2][0]],
		[dst_points[2][1]]
	]
	res = linalg.solve(x, y)
	return {'a': res[:4].reshape(2,2), 'b': res[4:]}
	
''' transform y = ax + b '''
def linearTransform(point, transform):
	''' calculate result '''
	[[res_x], [res_y]] = np.rint(np.dot(transform['a'], [[point[0]], [point[1]]]) + transform['b']).astype(int)
	return [res_x, res_y]
	
''' transform x = a^-1(y-b) '''
def linearTransformInv(point, transform):
	''' calculate inverse matrix '''
	inv = linalg.inv(transform['a'])
	''' calculate result '''
	[[res_x], [res_y]] = np.rint(np.dot(inv, ([[point[0]], [point[1]]] - transform['b']))).astype(int)
	return [res_x, res_y]
	
''' up_sampling '''
def up_sampling(src_img, dst_img, transform, area):
	''' calculate destination area '''
	dst_area = [linearTransform(area[0], transform), linearTransform(area[2], transform)]
	''' loop in destination area'''
	for dst_y in np.arange(dst_area[0][1], dst_area[1][1]):
		for dst_x in np.arange(dst_area[0][0], dst_area[1][0]):
			''' calculate original point '''
			[x, y] = linearTransformInv([dst_x, dst_y], transform)
			dst_img[dst_y][dst_x] = src_img[y][x]
	return dst_img

''' define path and name '''
path = '/Users/yckao85/Desktop/node_quick/'
src_name = '000044.jpg'
dst_name = 'destination.jpg'

''' read file and generate black result '''
src_img = io.imread(path + src_name)
dst_img = np.zeros(src_img.shape)


''' get width and height '''
width = src_img.shape[1]
height = src_img.shape[0]

''' pack source points '''
src_points = [
	[0, 0],
	[830, 630],
	[1595, 1760],
	[1100, 1050],
	[1340, 1430],
	[width, height]
]

''' pack destination points '''
dst_points = [
	[0, 0],
	[200, 200],
	[(width / 2 - 200) / 2, (height / 2 - 200) / 2],
	[width - (width / 2 - 200) / 2, height - (height / 2 - 200) / 2],
	[width - 200, height - 200],
	[width, height]
]


''' generate areas '''
src_areas = split_area_by_point(src_points)
dst_areas = split_area_by_point(dst_points)

''' generate transforms '''
transforms = []
for i in range(0, len(src_areas)):
	transforms.append(solve(src_areas[i], dst_areas[i]))

''' sampling new image'''
for i in range(0, len(src_areas)):
	print(i + 1, '/', len(src_areas))
	dst_img = up_sampling(src_img, dst_img, transforms[i], src_areas[i])

''' save '''
io.imsave(path + dst_name, dst_img.astype(np.uint8))