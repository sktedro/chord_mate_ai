import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pydub.playback import play
from pydub import AudioSegment 

# TODO
dsPath = "/media/tedro/#2/a_only_on_external/d_projekty/ai/chord_mate_ai/datasets/nsynth"


def filterDs(x, instrument, label, source, pitch):
    #  return tf.equal(x["instrument"]["family"], instrument) and tf.equal(x["instrument"]["label"], label)
    return ((tf.equal(x["instrument"]["family"], instrument) or instrument == -1)
            and (tf.equal(x["instrument"]["label"], label) or label == -1)
            and (tf.equal(x["instrument"]["source"], source) or label == -1)
            and (tf.equal(x["pitch"], pitch) or label == -1)
            )

def main():
    # TODO Choose a target
    # Amounts:
    # train: 60,788
    # test:  8,518
    # valid: 17,469

    target = "test"
    #  target = "train"


    # Load the dataset
    ds = tfds.load("nsynth", split=target, data_dir=dsPath, shuffle_files=True)

    # TODO:

    # Choose instrument family (11 classes)
    # 0 bass
    # 1 brass
    # 2 flute
    # 3 guitar
    # 4 keyboard
    # 5 mallet
    # 6 organ
    # 7 reed
    # 8 string
    # 9 synth_lead
    # 10 vocal
    family = 3

    # Choose instrument label (1006 classes)
    label = -1

    # Choose instrument source (3 classes)
    # 0 acoustic
    # 1 electronic
    # 2 synthetic
    source = 0

    # Choose pitch (128 classes)
    pitch = 56

    filteredDs = ds.filter(lambda x: filterDs(x, family, label, source, pitch))

    filteredData = filteredDs.take(1)
    print(filteredData)
    for note in filteredData:
        print(note)
        rawSignal = note["audio"].numpy()

        audio = AudioSegment(
            rawSignal.tobytes(),
            frame_rate=16000,
            sample_width=rawSignal.dtype.itemsize,
            channels=1
        )
        play(audio)


if __name__ == "__main__":
    main()
