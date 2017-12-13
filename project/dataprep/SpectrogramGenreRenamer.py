import sys
import os
from glob import glob


# Rename all pngs within the folder (recursive) with the given genre
# The spectrogram png names are of the form genre_artist_title.png
def rename_with_genre(folder_path, genre):
    print("Renaming all pngs with genre {} under {}".format(genre, folder_path))
    # Get all png files recursively
    pngs = glob(folder_path + '/**/*.png', recursive=True)
    counter = 1
    for png_file_path in pngs:
        print("Renaming file {} of {} - {}".format(counter, str(len(pngs)), png_file_path))
        rename_file_with_genre(png_file_path, genre)
        counter = counter + 1


def rename_file_with_genre(file_path, genre):
    # The spectrogram png names are of the form genre_artist_title.png
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    name_split = os.path.splitext(file_name)[0].split("_")
    new_name = "{}_{}_{}.png".format(genre, name_split[1], name_split[2])
    new_path = os.path.join(file_dir, new_name)
    os.rename(file_path, new_path)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: XXX.python [path to folder] [genre]")
        sys.exit(1)

    folder_path = sys.argv[1]
    genre = sys.argv[2]

    rename_with_genre(folder_path, genre)
