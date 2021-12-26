import sys
import numpy as np
import subprocess
from scipy.io import wavfile
import matplotlib.pyplot as plt
import struct


##############
## SETTINGS ##
##############

notesPath = "./audio/notes"
chordsPath = "./audio/chords"

# If set to true, orders of notes will be also picked randomly
# (the generated chord could consist of A0, B7, for example)
mixOrders = False

# If set to true, instruments playing notes will be also picked randomly
# (the generated chord could consist of A2 played by a guitar and B2 played by
# a piano)
mixInstruments = False


######################
## GLOBAL VARIABLES ##
######################

notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"] * 2


###############
## FUNCTIONS ##
###############


def printUsage():
    print("Usage: ")
    print("python3 gen_training_data.sh amount [targets]")
    print("")
    print("Amount stands for:")
    print("  The amount of .wav samples (chords) to generate")
    print("Targets can be (separated by spaces):")
    print("  major, minor, 7, 5, dim, dim7, aug, sus2, sus4, maj7, m7, 7sus4")
    print("")
    print("Notes are expected in ./audio/notes/instrument/")
    print("  While instrument can be any string")
    print("  Example of a note file: C#0_x.wav, while x goes from 0")
    print("Chords will be saved to ./audio/chords/instrument/")
    print("  Example of a chord file: C7_x.wav, while x goes from 0")

def handleArguments():
    args = sys.argv[1: ]

    if len(args) == 0:
        print("Please specify the amount")
        printUsage()
        quit(1)

    if args[0] == "--help" or args[0] == "-h":
        printUsage()
        quit(0)

    amount = int(args[0])

    possibleTargets = ["major", "minor", "7", "5", "dim", "dim7", "aug", "sus2", "sus4", "maj7", "m7", "7sus4"]
    if len(args) > 1:
        targets = args[1: ]
        for target in targets:
            if not target in possibleTargets:
                print("Wrong target:", target)
                printUsage()
                quit(1)
    else:
        targets = possibleTargets

    return amount, targets

def pickChord(chords, targets):
    chord = chords[np.random.randint(len(chords))]
    target = targets[np.random.randint(len(targets))]
    if target == "major":
        target = ""
    elif target == "minor":
        target = "m"
    return chord, target


def ls(path):
    return subprocess.run(["ls", path], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]

def getPitchClassesOfNotesInChord(chordType):
    # Major (C)
    if chordType == "":
        return [0, 4, 7]

    # Minor (Cm)
    elif chordType == "m":
        return [0, 3, 7]

    # Dominant seventh (C7)
    elif chordType == "7":
        return [0, 4, 7, 10]

    # Power chord (fifth chord) (C5)
    elif chordType == "5":
        return [0, 7]

    # Diminished chord (Cdim)
    elif chordType == "dim":
        return [0, 3, 6]

    # Diminished seventh chord (Cdim7)
    elif chordType == "dim7":
        return [0, 3, 6, 9]

    # Augmented chord (Caug, C+)
    elif chordType == "aug":
        return [0, 4, 8]

    # Suspended second chord (Csus2)
    elif chordType == "sus2":
        return [0, 2, 7]

    # Suspended fourth chord (Csus4)
    elif chordType == "sus4":
        return [0, 5, 7]

    # Major seventh (Cmaj7)
    elif chordType == "maj7":
        return [0, 4, 7, 11]

    # Minor seventh chord (Cm7, C-7)
    elif chordType == "m7":
        return [0, 3, 7, 10]

    # Seventh suspended chord (C7sus4)
    # TODO this might be wrong!
    elif chordType == "7sus4":
        return [0, 5, 7, 10]


def getNotesInChord(chord, chordType, order):

    rootIndex = notes.index(chord)

    pitchClasses = getPitchClassesOfNotesInChord(chordType)

    outputNotes = []

    for pc in pitchClasses:
        # Get the note index in notes arr
        index = rootIndex + pc

        # Get the note order
        noteOrder = order
        if index >= 12:
            noteOrder += 1

        # Now we can compose the note
        outputNotes.append(notes[index] + str(noteOrder))

    print("Notes to generate: ", outputNotes)
    return outputNotes

def getNoteSynonyms(note):
    order = note[-1]
    note = note[0: -1]
    noteIndexLeft = notes.index(note)
    noteIndexRight = noteIndexLeft + 12
    output = []
    # eg. for input B: Append B
    output.append(notes[noteIndexLeft] + order)
    # Append A##
    output.append(notes[noteIndexRight - 1] + "#" + order)
    # Append Cb
    output.append(notes[noteIndexLeft + 1] + "b" + order)

    # Discard notes containing ## and #b
    output = [n for n in output if "##" not in n and "#b" not in n]
    # Discard duplicates
    output = list(dict.fromkeys(output))
    return output

def chooseNotesFiles(useNotesFiles):
    return [synonyms[0] for synonyms in useNotesFiles]


##########
## MAIN ##
##########

# TODO

# If mixOrders == False, randomly pick a note order and filter out
# all notes that cannot be used in the final chord

def main():

    amount, targets = handleArguments()

    # Get all instruments (folders in notesPath)
    instruments = ls(notesPath)
    instruments.remove("0_short")

    # For amount
    for i in range(amount):

        # Get all notes for each instrument (.wav files in notesPath/instrument)
        notesFiles = []
        for instrument in instruments:
            notesFiles.append(ls(notesPath + "/" + instrument))


        # Randomly pick a chord (from len(targets) * len(notes) combinations)
        plainChord, chordType = pickChord(notes, targets)
        chord = plainChord + chordType
        print("Chord picked: ", chord)

        # Randomly pick an instrument
        instrument = instruments[np.random.randint(len(instruments))]
        print("Instrument picked: ", instrument)

        # Randomly pick an order (up to 7 - excluding 8)
        order = np.random.randint(0, 8)
        order = np.random.randint(2, 5) # TODO remove

        # Get a list of notes contained in the picked chord
        useNotes = getNotesInChord(plainChord, chordType, order)

        # Filter the notes files to ones of the instrument picked
        useNotesFiles = notesFiles[instruments.index(instrument)]

        # !! Gb is the same as F# and so on! It needs to be accepted
        useNotes = [getNoteSynonyms(note) for note in useNotes]

        # Filter the note files to ones contained in the picked chord
        useNotesFilesBackup = useNotesFiles
        useNotesFiles = []
        for noteSynonyms in useNotes:
            actNoteFiles = []
            for note in noteSynonyms:
                for noteFile in useNotesFilesBackup:
                    if note in noteFile:
                        actNoteFiles.append(noteFile)
            if len(actNoteFiles):
                useNotesFiles.append(actNoteFiles)
        #  print("Files to choose from: ", useNotesFiles)

        # If it is a mp3 file, check if there is a wav with the same name. If
        # not, convert it to a wav file with the same name
        useNotesFilesBackup = useNotesFiles
        useNotesFiles = []
        for actNotesFiles in useNotesFilesBackup:
            actNotesFilesBackup = actNotesFiles
            actNotesFiles = []
            for fileName in actNotesFilesBackup:
                if ".mp3" in fileName:
                    if not fileName.replace(".mp3", ".wav") in actNotesFilesBackup:
                        # Convert the mp3 to wav
                        path = notesPath + "/" + instrument + "/" + fileName
                        print("Converting", fileName, "to wav")
                        subprocess.call(["ffmpeg", "-loglevel", "error",
                                "-i", path, path.replace(".mp3", ".wav")])
                    actNotesFiles.append(fileName.replace(".mp3", ".wav"))
                else:
                    actNotesFiles.append(fileName)
            useNotesFiles.append(actNotesFiles)

        useNotesFiles = chooseNotesFiles(useNotesFiles)

        # Read the files and combine them into one
        signals = []
        sampleRates = []
        minLen = float("inf")
        path = notesPath + "/" + instrument + "/"
        for fileName in useNotesFiles:
            sampleRate, signal = wavfile.read(path + fileName)
            signals.append(signal)
            sampleRates.append(sampleRate)
            if float(len(signal)) < minLen:
                minLen = len(signal)

        # TODO Check if all sampleRates match

        # Sum the signals and trim it by the minimum length of the files
        signal = np.array(sum(signals)[0: int(minLen)])
        
        # Also make it exactly in a range of a short int
        multiplier = int((32767 / 2) / np.ndarray.max(np.absolute(signal)))
        signal = np.multiply(signal, multiplier)

        #  plt.plot(signal)
        #  plt.show()

        # Figure out a name for the file 
        # (they should be named chord_C_0.wav, chord_C_1.wav, chord_C#m_0.wav, ...)
        fileName = "chord_" + chord + "_" + instrument + "_"

        # If there is no file with that name, the number following should be
        # zero. Otherwise, just add 1
        existingFiles = ls(chordsPath)
        existingFiles = [f for f in existingFiles if fileName in f]
        if len(existingFiles) == 0:
            fileName += "0"
        else:
            nums = [f.split(".")[0].split("_")[-1] for f in existingFiles]
            highestNum = max([int(num) for num in nums])
            fileName += str(highestNum + 1)

        # Create chordsPath dir if it does not exist
        subprocess.call(["mkdir", "-p", chordsPath])

        # Write the signal to a file
        wavfile.write(chordsPath + "/" + fileName + ".wav", sampleRate, signal)
        print("Chord generated to", chordsPath + "/" + fileName + ".wav")

if __name__ == "__main__":
    main()
