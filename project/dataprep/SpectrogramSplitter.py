import sys
import os
import math
from glob import glob
from PIL import Image


def split_spectrogram(file_path, output_folder):
    file_name = os.path.basename(file_path)
    file_name_split = file_name.split("_")
    genre = file_name_split[0]

    # Create genre subfolder if needed
    success, full_output_folder = create_genre_folder(output_folder, genre)
    if not success:
        print("Could not create output folder, skipping file: " + file_path)
        return

    # We want 128x128 splits of the spectrogram
    # The spectrograms are of size Wx129, where W depends on the length of the song
    # 129 is one more than a power of two (128), SoX gives us that automatically for computational efficiency reasons
    split_dimensions = 128
    spectrogram = Image.open(file_path)
    width, height = spectrogram.size
    # Let's do floor here so we skip the last part of the song, which is often just silence
    number_of_splits = int(math.floor(width/split_dimensions))

    for split_count in range(number_of_splits):
        x_index = split_count * split_dimensions
        # (left, upper, right, lower)
        # Left is the x_index, upper is 1 since we want to cut away the top pixel in the 129 pixel tall image
        # right is the width range we want, lower is the index of the lowest pixel we want (counting from the top)
        # which is 129 in our case
        image_split = spectrogram.crop((x_index, 1, x_index + split_dimensions, height))
        image_split.save(os.path.join(full_output_folder, get_split_file_name(file_name, split_count)))


def get_split_file_name(spectrogram_name, number):
    # The spectrogram image names are of the form genre_artist_title.png OR genre_number.png
    # Return a new name of the form genre_artist_title_number.png OR genre_number_number.png
    # Genre, artist and title together with the number should form a unique name for the split file
    # If there are two instances of the same song by the same artist we do not care.
    # Remove file extension and split by underscore
    name_split = os.path.splitext(spectrogram_name)[0].split("_")
    if len(name_split) == 3:
        # name is of the form genre_artist_title.png
        return "{}_{}_{}_{}.png".format(name_split[0], name_split[1], name_split[2], number)
    elif len(name_split) == 2:
        # name is of the form genre_number.png
        return "{}_{}_{}.png".format(name_split[0], name_split[1], number)


def create_genre_folder(output_folder, genre):
    full_folder_path = os.path.join(output_folder, genre + "/")
    if not os.path.exists(os.path.dirname(full_folder_path)):
        try:
            print("Creating folder: " + full_folder_path)
            os.makedirs(os.path.dirname(full_folder_path))
            return True, full_folder_path
        except OSError as exc:
            print(exc)
            print("Could not create folder: " + full_folder_path)
            return False, full_folder_path
    else:
        return True, full_folder_path


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: XXX.python [path to folder or file] [output folder for spectrograms]")
        sys.exit(1)

    path = sys.argv[1]
    output_folder = sys.argv[2]

    print("Processing spectrogram/grams under: " + path)

    if os.path.isfile(path):
        split_spectrogram(path, output_folder)
    elif os.path.isdir(path):
        # See: https://stackoverflow.com/a/40755802
        spectrograms = glob(path + '/**/*.png', recursive=True)
        counter = 1
        for spectrogram_path in spectrograms:
            print("Processing spectrogram " + str(counter) + " of " + str(len(spectrograms))
                  + " - " + os.path.basename(spectrogram_path))
            split_spectrogram(spectrogram_path, output_folder)
            counter = counter + 1