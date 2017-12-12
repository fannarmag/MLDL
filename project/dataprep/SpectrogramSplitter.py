import sys
import os
from glob import glob


def split_spectrogram(file_path, output_folder):
    file_name = os.path.basename(file_path)
    file_name_split = file_name.split("_")
    genre = file_name_split[0]
    print(genre)


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
            print("")
            print("Processing spectrogram " + str(counter) + " of " + str(len(spectrograms)))
            split_spectrogram(spectrogram_path, output_folder)
            counter = counter + 1