import sys
import numpy as np
import subprocess



##############
## SETTINGS ##
##############

notesPath = "../audio/notes"
chordsPath = "../audio/chords"

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
    return chord + target



##########
## MAIN ##
##########

def main():

    amount, targets = handleArguments()
    chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Get all paths to all the .wav note files
    instrumentsCommand = ("ls " + notesPath).split(" ")
    instruments = subprocess.run(instrumentsCommand, stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]

    # For amount
    for i in range(amount):

        # Randomly pick a chord (from len(targets) * len(chords) combinations)
        #  chord = chords[np.random.randint(len(chords))] + targets[np.random.randint(len(targets))]
        chord = pickChord(chords, targets)
        print("Generating", chord)

        # Randomly pick an instrument
        instrument = instruments[np.random.randint(len(instruments))]
        print("Instrument picked: ", instrument)

        # Filter the note files to ones contained in the picked chord

        # Sort the note files into arrays based on their notes

        # If mixOrders == False, randomly pick a note order and filter out
        # all notes that cannot be used in the final chord

        # Read the files and combine them into one

        # Write it to a file (they should be named chord_C_1.wav, chord_C_2.wav, 
        # chord_C#m_1.wav, ...)



    # TODO don't forget that the output length must be the min length of inputs

if __name__ == "__main__":
    main()
