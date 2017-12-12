import sys
import eyed3
from glob import glob
from collections import defaultdict
from operator import add


# Collect statistics on what genres are represented in the mp3s in the folder
# and how many songs of each genre there are
def get_genre_stats(folder_path):
    print("Getting stats for folder: " + folder_path)
    # Get all mp3 files recursively
    # See: https://stackoverflow.com/a/40755802
    mp3s = glob(folder_path + '/**/*.mp3', recursive=True)
    print("Number of mp3 files: " + str(len(mp3s)))
    genre_dict = defaultdict(int)
    print("Collecting genre statistics")
    counter = 0
    hundred_counter = 1
    for mp3path in mp3s:
        counter = counter + 1
        if counter == 100:
            print("Processing file " + str(hundred_counter) + "00 of " + str(len(mp3s)))
            hundred_counter = hundred_counter + 1
            counter = 0
        audio_file = eyed3.load(mp3path)
        genre_dict[get_genre(audio_file)] = add(genre_dict[get_genre(audio_file)], 1)
    print_genre_dict(genre_dict)


def print_genre_dict(x):
    print("Genre statistics")
    for key, value in x.items():
        print(str(key) + ": " + str(value))


def get_genre(audio_file):
    if not audio_file.tag:
        return None
    elif not audio_file.tag.genre:
        return None
    else:
        return audio_file.tag.genre.name


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: XXX.python [path to folder]")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Increase eyed3 log level
    eyed3.log.setLevel("ERROR")
    get_genre_stats(folder_path)
