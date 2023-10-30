def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


my_image = "lenna.png"

import os
cwd = os.getcwd()

image_path = os.path.join(cwd, my_image)


# Loading Images in Python
from PIL import Image

image = Image.open(my_image)
type(image)

image.show()

# Viewing an image using matplotlib

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# Loading via path
image = Image.open(image_path)

print(image.size) #pixels
print(image.mode) #RGB or L(GrayScale)

im = image.load() # Puts the image into the computer memory

x = 0
y = 1
im[y,x]

image.save("lenna.jpg")

## Grayscale Images, Quantization and Color Channels  

from PIL import ImageOps 

image_gray = ImageOps.grayscale(image)

# Quantizing (Changing Intensity)
image_gray.quantize(256 // 2)

#get_concat_h(image_gray,  image_gray.quantize(256//2)).show(title="Lena") 
for n in range(3,8):
    plt.figure(figsize=(10,10))

    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
    plt.show()

# Color Channels

baboon = Image.open('baboon.png')
red, green, blue = baboon.split()
get_concat_h(baboon, red)
get_concat_h(baboon, blue)
get_concat_h(baboon, green)

## PIL Images into NumPy Arrays

import numpy as np

array= np.asarray(image)
print(type(array))

array = np.array(image)

# summarize shape
print(array.shape)

print(array)

array[0, 0]


## Indexing

plt.figure(figsize=(10,10))
plt.imshow(array)
plt.show()


rows = 256
plt.figure(figsize=(10,10))
plt.imshow(array[0:rows,:,:])
plt.show()

columns = 256
plt.figure(figsize=(10,10))
plt.imshow(array[:,0:columns,:])
plt.show()

A = array.copy()
plt.imshow(A)
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array)
plt.show()

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array[:,:,0], cmap='gray')
plt.show()

baboon_red=baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.show()

baboon_blue=baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()


