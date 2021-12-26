import sys
import numpy as np
import subprocess



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
    print("  Example of a note file: C#0_x.wav, while x goes from 1")
    print("Chords will be saved to ./audio/chords/instrument/")
    print("  Example of a chord file: C7_x.wav, while x goes from 1")

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
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"] * 2

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


##########
## MAIN ##
##########

def main():

    amount, targets = handleArguments()
    chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Get all instruments (folders in notesPath)
    instruments = ls(notesPath)

    # Get all notes for each instrument (.wav files in notesPath/instrument)
    notesFiles = []
    for instrument in instruments:
        notesFiles.append(ls(notesPath + "/" + instrument))
    # Filter out ones that don't end with .wav
    #  for i in range(len(notesFiles)):
        #  notesFiles[i] = [note for note in notesFiles[i] if ".wav" in note]




    # For amount
    for i in range(amount):

        # Randomly pick a chord (from len(targets) * len(chords) combinations)
        #  chord = chords[np.random.randint(len(chords))] + targets[np.random.randint(len(targets))]
        plainChord, chordType = pickChord(chords, targets)
        chord = plainChord + chordType
        print("Chord picked: ", chord)

        # Randomly pick an instrument
        instrument = instruments[np.random.randint(len(instruments))]
        print("Instrument picked: ", instrument)

        # TODO Randomly pick an order (up to 7 - excluding 8)
        order = 2

        # Get a list of notes contained in the picked chord
        useNotes = getNotesInChord(plainChord, chordType, order)

        # Filter the note files to ones contained in the picked chord
        useNotesFiles = notesFiles[instruments.index(instrument)]

        # Sort the note files into arrays based on their notes

        # If mixOrders == False, randomly pick a note order and filter out
        # all notes that cannot be used in the final chord

        # Read the files and combine them into one

        # Write it to a file (they should be named chord_C_1.wav, chord_C_2.wav, 
        # chord_C#m_1.wav, ...)



    # TODO don't forget that the output length must be the min length of inputs

if __name__ == "__main__":
    main()
