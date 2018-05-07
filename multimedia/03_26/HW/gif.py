from PIL import Image, ImageChops

# read image
base_image = Image.open('./base.jpg')
overlay_image = Image.open('./overlay.jpg')

# resize
base_image = base_image.resize((600, 450))
overlay_image = overlay_image.resize((600, 450))

# define result
result = []

# process 50 frames
for _ in range(0, 50):
    # add image using ImageChops.add
    added_image = ImageChops.add(base_image, overlay_image)
    # move image using ImageChops.offset
    overlay_image = ImageChops.offset(overlay_image, -3, 5)
    result.append(added_image)

# save
result[0].save('result.gif', save_all = True, append_images = result[1:], duration = 500, loop = 1, optimize = True)
