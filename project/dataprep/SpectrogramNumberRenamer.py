import sys
import os
from glob import glob
from collections import defaultdict


# Rename all pngs within the folder (recursive)
# The spectrogram png names are of the form genre_artist_title.png OR genre_number.png (already renamed files)
# Rename all of them to genre_number.png, where the number is the number of the song within that genre
def rename_spectrograms(folder_path):
    print("Renaming all pngs under {}".format(folder_path))
    # Get all png files recursively
    pngs = glob(folder_path + '/**/*.png', recursive=True)
    genre_dict = build_genre_dict(pngs)
    for genre_key in genre_dict.keys():
        print("Processing {}...".format(genre_key))
        rename_spectrograms_of_genre(genre_dict[genre_key])


# Takes a list of file paths to pngs and sorts them in a dictionary of lists by genre
def build_genre_dict(pngs):
    d = defaultdict(list)
    for png_file_path in pngs:
        file_name = os.path.basename(png_file_path)
        # File name is of the form genre_artist_title_number.png or genre_number.png
        genre = file_name.split("_")[0]
        d[genre].append(png_file_path)
    return d


def rename_spectrograms_of_genre(png_paths):
    counter = 0
    for png_path in png_paths:
        rename_file_with_number(png_path, counter)
        counter = counter + 1


def rename_file_with_number(file_path, number):
    # The spectrogram png names are of the form genre_artist_title.png
    # OR genre_number.png
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    name_split = os.path.splitext(file_name)[0].split("_")
    new_name = "{}_{}.png".format(name_split[0], number)
    new_path = os.path.join(file_dir, new_name)
    print("Renaming {} to {}".format(file_path, new_path))
    os.rename(file_path, new_path)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: XXX.python [path to folder]")
        sys.exit(1)

    folder_path = sys.argv[1]

    rename_spectrograms(folder_path)
