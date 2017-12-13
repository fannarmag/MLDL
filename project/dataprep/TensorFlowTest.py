import tensorflow as tf
import numpy as np
from PIL import Image  # from Pillow

label_dict = {
    'Classical': 0,
    'Electronic': 1,
    'Pop': 2,
    'HipHop': 3,
    'Metal': 4,
    'Rock': 5
}

# Just a small test file to play around with tensorflow locally on our dataset

im = Image.open("spectrograms/Metal_Cobalt_Gin.png")
# im.filename is spectrogram/file.png - extract the image filename
filename = im.filename.split("/")[len(im.filename.split("/")) - 1]
genre = filename.split("_")[0]
print(genre, im.size)
# img.show()

# if we can't extract the label from the image we should not train on it
if genre in label_dict:
    value = label_dict.get(genre)
    print(str(genre) + " - " + str(value))
    ohv = tf.one_hot(value, label_dict.__len__())
    with tf.Session():
        print(ohv.eval())

num_input = 128*128*1

img = im.resize((128, 128), resample=Image.ANTIALIAS)
imgData = np.asarray(img, dtype=np.uint8).reshape(128, 128, 1)
imgData = imgData / 255 - 0.5

print(imgData)


#with open("spectrograms/Metal_Cobalt_Gin.png", "rb") as imageFile:
#  f = imageFile.read()
#  b = bytearray(f)

#image = tf.decode_raw(b, tf.uint8)
#image.set_shape([num_input])
#image = tf.reshape(image, [128, 128, 1], tf.float32)

# Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
#image = tf.cast(image, tf.float32) / 255 - 0.5

# print(image[0])

