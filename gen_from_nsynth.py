from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pydub.playback import play
from pydub import AudioSegment 
import process_audio 
import generate_chords
import sys

# !!!
# The problem is that filtering (and getting only the wanted data) takes a very
# long time
# So.. this script is far from finished
# Maybe I shouldn't even use this, or use this only for generating notes, not
# chords

# TODO
dsPath = "/media/tedro/#2/a_only_on_external/d_projekty/ai/chord_mate_ai/datasets/nsynth"

def filterDs(x, instrument, label, source, pitch):
    return (
            (instrument == -1 or tf.equal(x["instrument"]["family"], instrument))
        and (label == -1 or  tf.equal(x["instrument"]["label"], label))
        and (source == -1 or tf.equal(x["instrument"]["source"], source))
        and (pitch == -1 or (x["pitch"] > pitch[0] and x["pitch"] < pitch[-1]))
    )

#  class Chord:
    #  def __init__(self, instrument, label, source, basePitch):
        #  self.instrument = instrument
        #  self.source = source
        #  self.label = label
        #  self.basePitch = basePitch
        #  self.complete = False
        #  self.noteIndicesToUse = generate_chords.getPitchClassesOfNotesInChord(chordType)
        #  self.signal = []

    #  def checkMatch(instrument, label, source, pitch):
        #  if (
                    #  pitch - basePitch in self.noteIndicesToUse
                #  and self.instrument == instrument
                #  and self.complete == False
                #  ):

    #  def addSignal(sig):
        #  self.signal = (np.array(self.signal) + np.array(sig)) / 2




"""
Inputs: 
Desired chord / chords (index)
Amount of chord / chords
"""
def main():
    # TODO Choose a target
    # Amounts in targets:
    # train: 289,205
    # test:  4,096
    # valid: 12,678

    #  target = "train"
    target = "test"
    #  target = "valid"

    try:
        amount = int(sys.argv[1])
        print("Amount selected:", amount)
    except:
        print("Please specify the amount of chords")
        quit(0)

    # Load the dataset
    ds = tfds.load("nsynth", split=target, data_dir=dsPath, shuffle_files=True)
    print("Dataset loaded")

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
    reqInstrument = -1
    reqInstrument = 3

    # Choose instrument label (1006 classes)
    # We probably won't, ever - label seems to be unique in the whole dataset
    # for an instrument family (eg. keyboard_electronic_054 has label 480)
    reqLabel = -1

    # Choose instrument source (3 classes)
    # 0 acoustic
    # 1 electronic
    # 2 synthetic
    reqSource = -1
    #  reqSource = 0

    # Choose pitch (128 classes) - 0 = C-1, 12 = C0, C1 = 24, ...
    reqPitch = -1
    reqPitch = [36, 48]

    filteredDs = ds.filter(lambda x: filterDs(x, reqInstrument, reqLabel, reqSource, reqPitch))

    # Note: audio from the dataset is in range -1 to 1

    for note in filteredDs.take(1):

        sampleRate = 16000

        # Get the first note
        noteId = note["id"].numpy()
        pitch = note["pitch"].numpy()
        instrument = note["instrument"]["family"].numpy()
        label = note["instrument"]["label"].numpy()
        source = note["instrument"]["source"].numpy()
        rawSignal = note["audio"].numpy()

        print("Using a note", note["id"].numpy(), 
                "with pitch", note["pitch"].numpy(), 
                "played by instrument", note["instrument"]["family"].numpy())

        # TODO Pick a chord type (major, minor, 7, ...) - there are 12 types
        chordType = ""

        # Get what notes to use
        noteIndicesToUse = generate_chords.getPitchClassesOfNotesInChord(chordType)

        print("Using relative note indices:", noteIndicesToUse)

        notes = []
        notes.append(note)

        #  for noteIndexToUse in noteIndicesToUse[1: ]:
            #  actFilteredDs = ds.filter(lambda x: filterDs(x, instrument, -1, source, [pitch + noteIndexToUse]))
            #  for n in actFilteredDs.take(1):
                #  notes.append(n)

        jointSignal = []
        minLen = -1
        for actNote in notes:
            print("Using a note", actNote["id"].numpy(), 
                    "with pitch", actNote["pitch"].numpy(), 
                    "played by", actNote["instrument"]["family"].numpy())

            sig = actNote["audio"].numpy()

            # Crop the signal by loudness (remove the quiet part)
            croppedSig = process_audio.cropByLoudness(sig, verbose = False)

            if minLen == -1 or len(croppedSig) < minLen:
                minLen = len(croppedSig)

            if len(jointSignal) == 0:
                jointSignal = croppedSig
            else:
                jointSignal = jointSignal + croppedSig
            #  print(len(jointSignal))

        jointSignal = jointSignal / len(notes)
        jointSignal = jointSignal[0: minLen]


        audio = AudioSegment(
            jointSignal.tobytes(),
            frame_rate=sampleRate,
            sample_width=jointSignal.dtype.itemsize,
            channels=1
        )
        play(audio)

    print("Done")


if __name__ == "__main__":
    main()
