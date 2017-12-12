import tensorflow as tf
from PIL import Image  # from Pillow

# Just a small test file to play around with tensorflow locally on our dataset

im = Image.open("spectrograms/Metal_Cobalt_Gin.png")
# im.filename is spectrogram/file.png - extract the image filename
filename = im.filename.split("/")[len(im.filename.split("/")) - 1]
genre = filename.split("_")[0]
print(genre, im.size)
# img.show()

label_dict = {
    'Classical': 0,
    'Electronic': 1,
    'Grindcore': 2,
    'HipHop': 3,
    'Metal': 4,
    'Rock': 5
}

# if we can't extract the label from the image we should not train on it
if genre in label_dict:
    value = label_dict.get(genre)
    print(str(genre) + " - " + str(value))
    ohv = tf.one_hot(value, label_dict.__len__())
    with tf.Session():
        print(ohv.eval())