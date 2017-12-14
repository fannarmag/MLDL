import sys
import os
from glob import glob


# Merge two folders of spectrogram files with names of the form genre_number.png
# The contents of folder 2 will be moved into folder 1, renaming the files from folder 2 so that their numbers are
# higher than the numbers of the files in folder 1
# This script assumes that all spectrograms within the two folders are of the same genre
def merge_folders(folder_path_1, folder_path_2):
    print("Merging folders {} and {}".format(folder_path_1, folder_path_2))
    # Get all png files recursively
    pngs1 = glob(folder_path_1 + '/**/*.png', recursive=True)
    pngs2 = glob(folder_path_2 + '/**/*.png', recursive=True)
    max_number_1 = get_highest_png_number(pngs1)
    # Move pngs2 to folder 1, so that their numbers are higher than the numbers of the pngs in folder 1
    new_number = max_number_1 + 1
    for png_file_path_to_move in pngs2:
        file_name = os.path.basename(png_file_path_to_move)
        # The spectrogram png names are of the form genre_number.png
        name_split = os.path.splitext(file_name)[0].split("_")
        new_name = "{}_{}.png".format(name_split[0], new_number)
        new_path = os.path.join(folder_path_1, new_name)
        print("Renaming/moving {} to {}".format(png_file_path_to_move, new_path))
        os.rename(png_file_path_to_move, new_path)
        new_number = new_number + 1


def get_highest_png_number(png_paths):
    max_number = -1
    for png_path in png_paths:
        file_name = os.path.basename(png_path)
        # File names are of the form genre_number.png
        number = int(os.path.splitext(file_name)[0].split("_")[1])
        if number > max_number:
            max_number = number
    return max_number


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: XXX.python [path to folder 1] [path to folder 2]")
        sys.exit(1)

    folder_path_1 = sys.argv[1]
    folder_path_2 = sys.argv[2]

    merge_folders(folder_path_1, folder_path_2)
