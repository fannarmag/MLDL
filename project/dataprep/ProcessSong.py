import os
from subprocess import Popen, PIPE, STDOUT
import eyed3


def main():
    file_path = "data/01.mp3"
    generate_spectrogram(file_path)


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


def generate_spectrogram(filepath):
    current_path = os.path.dirname(os.path.realpath(__file__))

    audio_file = eyed3.load(filepath)

    # Generate a mono version of the song if needed
    filepath_to_convert = filepath
    mono_track_created = False
    if not is_mono(audio_file):
        filepath_to_convert = generate_mono_version(filepath)
        mono_track_created = True

    output_file_path = os.path.join("spectrograms", get_spectrogram_name(audio_file))
    print("Generating spectrogram at: " + output_file_path)
    command = "sox {} -n spectrogram -Y 200 -X {} -m -r -o {}".format(filepath_to_convert, 50, output_file_path)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=current_path)
    output, errors = p.communicate()
    if errors:
        print(errors)

    if mono_track_created:
        # Remove temporary mono track
        print("Removing temp mono track at: " + filepath_to_convert)
        os.remove(filepath_to_convert)


def generate_mono_version(file_path):
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Generate output file path
    filename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)
    generated_file_path = os.path.join(dirname, "mono_" + filename)

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
    main()

