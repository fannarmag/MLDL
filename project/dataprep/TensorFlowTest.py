import tensorflow as tf
from PIL import Image  # from Pillow

# Just a small test file to play around with tensorflow locally on our dataset

img = Image.open("spectrograms/Metal_Cobalt_Gin.png")
print(img.format, img.size, img.mode)
# img.show()

image_contents = tf.read_file("spectrograms/Cobalt_Gin_Metal.png")
decoded_image = tf.image.decode_png(image_contents, channels=1, dtype=tf.uint8)
print(decoded_image.shape)
#resized_image = tf.image.resize_images(decoded_image, [129, 129])
#print(resized_image.shape)
#print(resized_image)



