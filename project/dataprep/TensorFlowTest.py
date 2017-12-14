import tensorflow as tf
import numpy as np
from PIL import Image  # from Pillow
import matplotlib.pyplot as plt
import glob

label_dict = {
    'Classical': 0,
    'Electronic': 1,
    'Pop': 2,
    'HipHop': 3,
    'Metal': 4,
    'Rock': 5
}


# Load images from folder and create list of tuples
# of the form: [(label, img_data), (label, img_data), ...]
def load_spectrograms(folder_path):
    print("Load spectrograms in " + folder_path + "...")
    data = []
    for file in glob.glob(folder_path + "/*.png"):
        im = Image.open(file)
        # extract label
        filename = im.filename.split("/")[len(im.filename.split("/")) - 1]
        genre = filename.split("_")[0]

        # if we can't extract the label from the image we should not train on it
        if genre not in label_dict:
            continue;

        label_val = label_dict.get(genre)
        label = tf.one_hot(label_val, label_dict.__len__())

        # print just for testing
        with tf.Session():
            print(label.eval())

        img = im.resize((128, 128), resample=Image.ANTIALIAS)
        img_data = np.asarray(img, dtype=np.uint8).reshape(128, 128, 1)
        img_data = (img_data / 255.) - 0.5  # for mean around 0

        # print(img_data)
        # plt.imshow(img_data, cmap='Greys')
        # plt.show()

        data.append((label, img_data))

    return data


data = load_spectrograms("spectrograms/HipHop/")
print(data)

# with open("spectrograms/HipHop/HipHop_2Pac_2 Of Amerikaz Most Wanted_0.png", "rb") as imageFile:
#  f = imageFile.read()
#  b = bytearray(f)

# image = tf.decode_raw(b, tf.uint8)
# image.set_shape([num_input])
# image = tf.reshape(image, [128, 128, 1], tf.float32)

# Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
# image = tf.cast(image, tf.float32) / 255 - 0.5

# print(image[0])

# def create_db():
#   train_batch = open("../train_batch_img",'wb')
#   train_batch_y = open("../train_batch_label",'wb')
#   for file in glob.glob("spectrograms/HipHop/*.png"):
#       imgx = Image.open(file)
#       img = np.asarray(img, dtype='float64') / 256.
#       pickle.dump(img,train_batch)
#       pickle.dump(folder_name,train_batch_y)
#   train_batch.close()
#   train_batch_y.close()

