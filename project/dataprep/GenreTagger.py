import sys
import eyed3
from glob import glob


# Tag all mp3s within the folder (recursive) with the given genre
def tag_with_genre(folder_path, genre):
    print("Tagging all mp3s as {} under {}".format(genre, folder_path))
    # Get all mp3 files recursively
    # See: https://stackoverflow.com/a/40755802
    mp3s = glob(folder_path + '/**/*.mp3', recursive=True)
    counter = 1
    for mp3_file_path in mp3s:
        print("Tagging file {} of {} - {}".format(counter, str(len(mp3s)), mp3_file_path))
        tag_song_with_genre(mp3_file_path, genre)
        counter = counter + 1


def tag_song_with_genre(file_path, genre):
    audiofile = eyed3.load(file_path)
    if audiofile.tag is None:
        audiofile.initTag()
    audiofile.tag.genre = genre
    audiofile.tag.save()


def get_genre(audio_file):
    if not audio_file.tag:
        return None
    elif not audio_file.tag.genre:
        return None
    else:
        return audio_file.tag.genre.name


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: XXX.python [path to folder] [genre]")
        sys.exit(1)

    folder_path = sys.argv[1]
    genre = sys.argv[2]

    # Increase eyed3 log level
    eyed3.log.setLevel("ERROR")
    tag_with_genre(folder_path, genre)
