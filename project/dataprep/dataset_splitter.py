import sys
import os
from glob import glob
from collections import defaultdict
import random


def split_data(folder_path, validation_ratio, testing_ratio):
    # Create dataset split folders
    _, training_folder = create_folder_with_name(folder_path, "training")
    _, validation_folder = create_folder_with_name(folder_path, "validation")
    _, testing_folder = create_folder_with_name(folder_path, "testing")
    png_paths = glob(folder_path + '/**/*.png', recursive=True)
    genre_dict = build_genre_dict(png_paths)
    # Create genre folders within dataset split folders
    for genre, _ in genre_dict.items():
        png_paths = genre_dict[genre]
        print("Found {} files for {}".format(str(len(png_paths)), genre))
        random.shuffle(png_paths)
        validation_count = int(len(png_paths) * validation_ratio)
        testing_count = int(len(png_paths) * testing_ratio)
        training_count = len(png_paths)-validation_count-testing_count
        validation_paths = png_paths[:validation_count]
        testing_paths = png_paths[validation_count:validation_count+testing_count]
        training_paths = png_paths[-training_count:]
        print("Splitting {} into {} validation, {} testing, {} training"
              .format(genre, str(len(validation_paths)), str(len(testing_paths)), str(len(training_paths))))

        # Validation
        _, genre_validation_folder = create_folder_with_name(validation_folder, genre)
        for path in validation_paths:
            move_file(path, genre_validation_folder)

        # Testing
        _, genre_testing_folder = create_folder_with_name(testing_folder, genre)
        for path in testing_paths:
            move_file(path, genre_testing_folder)

        # Training
        _, genre_training_folder = create_folder_with_name(training_folder, genre)
        for path in training_paths:
            move_file(path, genre_training_folder)


def move_file(file_path, destination_folder):
    file_name = os.path.basename(file_path)
    new_path = os.path.join(destination_folder, file_name)
    #print("Renaming/moving {} to {}".format(file_path, new_path))
    os.rename(file_path, new_path)


# Takes a list of file paths to pngs and sorts them in a dictionary of lists by genre
def build_genre_dict(pngs):
    d = defaultdict(list)
    for png_file_path in pngs:
        file_name = os.path.basename(png_file_path)
        # File name is of the form genre_artist_title_number.png or genre_number.png
        genre = file_name.split("_")[0]
        d[genre].append(png_file_path)
    return d


def create_folder_with_name(output_folder, name):
    full_folder_path = os.path.join(output_folder, name + "/")
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

    if len(sys.argv) != 2:
        print("Usage: XXX.python [path to folder]")
        sys.exit(1)

    folder_path = sys.argv[1]
    validation_ratio = 0.3
    testing_ratio = 0.1

    split_data(folder_path, validation_ratio, testing_ratio)