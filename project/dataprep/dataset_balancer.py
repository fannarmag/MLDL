import sys
import os
from glob import glob
import random

# Script that takes a random subset of size N of files from a folder and moves them into a new folder


def balance_data(folder_path, output_folder, desired_file_count):
    png_paths = glob(folder_path + '/*.png', recursive=True)
    print("Found {} files".format(str(len(png_paths))))
    if len(png_paths) <= desired_file_count:
        print("Don't have the desired file count, aborting")
        return

    _, output_folder = create_folder_with_name(output_folder, "!subset_" + str(desired_file_count))

    random.shuffle(png_paths)
    paths_to_move = png_paths[:desired_file_count]

    print("Moving {} random files to {}".format(str(len(paths_to_move)), output_folder))

    count = 1
    for path in paths_to_move:
        if count == 1 or count % 100 == 0:
            print("Moving file {} of {}".format(count, str(len(paths_to_move))))
        move_file(path, output_folder)
        count = count + 1


def move_file(file_path, destination_folder):
    file_name = os.path.basename(file_path)
    new_path = os.path.join(destination_folder, file_name)
    #print("Renaming/moving {} to {}".format(file_path, new_path))
    os.rename(file_path, new_path)


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

    if len(sys.argv) != 4:
        print("Usage: XXX.python [path to folder] [output folder] [desired file count]")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_folder = sys.argv[2]
    desired_file_count = int(sys.argv[3])

    print("Will create random subset of {} files from {}".format(str(desired_file_count), folder_path))

    balance_data(folder_path, output_folder, desired_file_count)
