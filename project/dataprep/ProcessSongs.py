import os
import sys
from subprocess import Popen, PIPE, STDOUT
import eyed3
from glob import glob


def is_mono(audio_file):
    return audio_file.info.mode == 'Mono'


def get_genre(audio_file):
    if not audio_file.tag.genre:
        return None
    else:
        return audio_file.tag.genre.name


def get_spectrogram_name(audio_file):
    return "{0!s}_{1!s}_{2!s}{3!s}"\
        .format(audio_file.tag.artist, audio_file.tag.title, get_genre(audio_file), ".png")


def generate_spectrogram(file_path, output_folder_path):
    current_path = os.path.dirname(os.path.realpath(__file__))

    audio_file = eyed3.load(file_path)

    print("Processing: " + audio_file.tag.artist + " - " + audio_file.tag.album + " - " + audio_file.tag.title)

    # Generate a mono version of the song if needed
    file_path_to_convert = file_path
    mono_track_created = False
    if not is_mono(audio_file):
        file_path_to_convert = generate_mono_version(file_path)
        mono_track_created = True

    output_file_path = os.path.join(output_folder_path, get_spectrogram_name(audio_file))
    print("Generating spectrogram at: " + output_file_path)
    command = "sox {} -n spectrogram -Y 200 -X {} -m -r -o {}".format(file_path_to_convert, 50, output_file_path)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=current_path)
    output, errors = p.communicate()
    if errors:
        print(errors)

    if mono_track_created:
        # Remove temporary mono track
        print("Removing temp mono track at: " + file_path_to_convert)
        os.remove(file_path_to_convert)


def generate_mono_version(file_path):
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Generate output file path
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    generated_file_path = os.path.join(dir_name, "mono_" + file_name)

    print("Generating mono file at: " + generated_file_path)
    command = "sox {} {} remix 1,2".format(file_path, generated_file_path)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=current_path)
    output, errors = p.communicate()
    if output:
        print(output)
    if errors:
        print(errors)
    return generated_file_path


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: XXX.python [path to folder or file] [output folder for spectrograms]")
        sys.exit(1)

    path = sys.argv[1]
    output_folder = sys.argv[2]

    if os.path.isfile(path):
        generate_spectrogram(path, output_folder)
    elif os.path.isdir(path):
        # See: https://stackoverflow.com/a/40755802
        mp3s = glob(path + '/**/*.mp3', recursive=True)
        counter = 1
        for mp3path in mp3s:
            print("Processing mp3 " + str(counter) + " of " + str(len(mp3s)))
            generate_spectrogram(mp3path, output_folder)
            counter = counter + 1


