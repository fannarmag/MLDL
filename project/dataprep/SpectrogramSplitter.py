import sys
import os
from glob import glob


def split_spectrogram(file_path, output_folder):
    file_name = os.path.basename(file_path)
    file_name_split = file_name.split("_")
    genre = file_name_split[0]
    success, full_output_folder = create_genre_folder(output_folder, genre)
    if not success:
        print("Could not create output folder, skipping file: " + file_path)
        return


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
            print("")
            print("Processing spectrogram " + str(counter) + " of " + str(len(spectrograms)))
            split_spectrogram(spectrogram_path, output_folder)
            counter = counter + 1