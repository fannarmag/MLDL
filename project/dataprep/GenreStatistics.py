import sys
import eyed3
import os
from glob import glob
from collections import defaultdict
from operator import add


# Collect statistics on what genres are represented in the mp3s or pngs in the folder
# and how many songs of each genre there are
def get_genre_stats(folder_path, file_type):
    print("Getting stats for {} in folder {}".format(file_type, folder_path))
    # Get all files recursively
    # See: https://stackoverflow.com/a/40755802
    file_paths = []
    if file_type == "mp3":
        file_paths = glob(folder_path + '/**/*.mp3', recursive=True)
        print("Number of mp3 files: " + str(len(file_paths)))
    elif file_type == "png":
        file_paths = glob(folder_path + '/**/*.png', recursive=True)
        print("Number of png files: " + str(len(file_paths)))

    print("Collecting genre statistics")
    genre_dict = defaultdict(int)
    counter = 0
    hundred_counter = 1
    for file_path in file_paths:
        counter = counter + 1
        if counter == 100:
            print("Processing file " + str(hundred_counter) + "00 of " + str(len(file_paths)))
            hundred_counter = hundred_counter + 1
            counter = 0
        if file_type == "mp3":
            audio_file = eyed3.load(file_path)
            genre_dict[get_genre_mp3(audio_file)] = add(genre_dict[get_genre_mp3(audio_file)], 1)
        elif file_type == "png":
            genre_dict[get_genre_png(file_path)] = add(genre_dict[get_genre_png(file_path)], 1)

    print_genre_dict(genre_dict)


def print_genre_dict(x):
    print("Genre statistics")
    for key, value in x.items():
        print(str(key) + ": " + str(value))


def get_genre_mp3(audio_file):
    if not audio_file.tag:
        return None
    elif not audio_file.tag.genre:
        return None
    else:
        return audio_file.tag.genre.name

def get_genre_png(file_path):
    file_name = os.path.basename(file_path)
    # png file names are of the form genre_....png
    genre = os.path.splitext(file_name)[0].split("_")[0]
    return genre


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: XXX.python [path to folder] [mp3/png]")
        sys.exit(1)

    folder_path = sys.argv[1]
    file_type = sys.argv[2]

    if file_type != "mp3" and file_type != "png":
        print("Usage: XXX.python [path to folder] [mp3/png]")
        sys.exit(1)

    # Increase eyed3 log level
    eyed3.log.setLevel("ERROR")
    get_genre_stats(folder_path, file_type)
