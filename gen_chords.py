import sys
import numpy as np
import subprocess
from scipy.io import wavfile
import matplotlib.pyplot as plt
import struct
import settings
import misc

######################
## GLOBAL VARIABLES ##
######################

notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"] * 2

###############
## FUNCTIONS ##
###############

def printUsage():
    print("Usage: ")
    print("python3 gen_training_data.sh amount instrumentsAmount [targets]")
    print("")
    print("Amount stands for:")
    print("  The amount of .wav samples (chords) to generate")
    print("InstrumentsAmount stands for:")
    print("  The amount of instruments to use when generating the chord")
    print("  (Each instrument plays the chord separately, but audio is merged)")
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

    if len(args) < 2:
        print("Please specify the amount of chords and amount of instruments")
        printUsage()
        quit(1)

    if args[0] == "--help" or args[0] == "-h":
        printUsage()
        quit(0)

    amount = int(args[0])
    instrumentsAmount = int(args[1])

    possibleTargets = ["major", "minor", "7", "5", "dim", "dim7", "aug", "sus2", "sus4", "maj7", "m7", "7sus4"]
    if len(args) > 2:
        targets = args[2: ]
        for target in targets:
            if not target in possibleTargets:
                print("Wrong target:", target)
                printUsage()
                quit(1)
    else:
        targets = possibleTargets

    return amount, instrumentsAmount, targets

def pickChord(chords, targets):
    chord = chords[np.random.randint(len(chords))]
    target = targets[np.random.randint(len(targets))]
    if target == "major":
        target = ""
    elif target == "minor":
        target = "m"
    return chord, target


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

# Pick randomly from files available containing the same note ('synonyms')
def chooseNotesFiles(useNotesFiles):
    return [synonyms[np.random.randint(len(synonyms))] for synonyms in useNotesFiles]

# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Dynamics.html
def compute_power(x, Fs):
    win_len_sec=0.05
    power_ref=10**(-12)
    """Computation of the signal power in dB

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        x (np.ndarray): Signal (waveform) to be analyzed
        Fs (scalar): Sampling rate
        win_len_sec (float): Length (seconds) of the window (Default value = 0.1)
        power_ref (float): Reference power level (0 dB) (Default value = 10**(-12))

    Returns:
        power_db (np.ndarray): Signal power in dB
    """
    win_len = round(win_len_sec * Fs)
    win = np.ones(win_len) / win_len
    power_db = np.log10(np.convolve(x**2, win, mode='same') / power_ref)
    return power_db

##########
## MAIN ##
##########

def main():

    print("==================================================")
    print("==================================================")

    amount, instrumentsAmount, targets = handleArguments()

    # Get all instruments (folders in settings.notesPath)
    instruments = misc.ls(settings.notesPath)
    if "0_short" in instruments:
        instruments.remove("0_short")

    # For amount
    for i in range(amount):

        print("==================================================")

        err = False

        # Randomly pick a chord (from len(targets) * len(notes) combinations)
        plainChord, chordType = pickChord(notes, targets)
        chord = plainChord + chordType
        print("Chord picked: ", chord)

        instrumentsUsed = []
        orders = []
        for i in range(instrumentsAmount):
            # Randomly pick an instrument
            instrumentsUsed.append(instruments[np.random.randint(len(instruments))])
            if not settings.mixInstruments:

                instrumentsUsed[i] = instrumentsUsed[0]

            # Randomly pick an order
            orders.append(np.random.randint(settings.minOrder, settings.maxOrder + 1))

        print("Instruments picked: ", instrumentsUsed)

        useNotesPaths = []
        for i in range(instrumentsAmount):

            # Get all notes for each instrument (.wav files in settings.notesPath/instrument)
            notesFiles = []
            for instrument in instruments:
                notesFiles.append(misc.ls(settings.notesPath + "/" + instrument))

            # Get a list of notes contained in the picked chord
            useNotes = getNotesInChord(plainChord, chordType, orders[i])

            # Filter the notes files to ones of the instrument picked
            useNotesFiles = notesFiles[instruments.index(instrumentsUsed[i])].copy()

            # !! Gb is the same as F# and so on! It needs to be accepted
            useNotes = [getNoteSynonyms(note) for note in useNotes].copy()

            # Filter the note files to ones contained in the picked chord
            useNotesFilesBackup = useNotesFiles.copy()
            useNotesFiles = []
            for noteSynonyms in useNotes:
                actNoteFiles = []
                for note in noteSynonyms:
                    for noteFile in useNotesFilesBackup:
                        if note in noteFile:
                            actNoteFiles.append(noteFile)
                if len(actNoteFiles):
                    useNotesFiles.append(actNoteFiles)

            # If it is a mp3 file, check if there is a wav with the same name. If
            # not, convert it to a wav file with the same name
            useNotesFilesBackup = useNotesFiles.copy()
            useNotesFiles = []
            for actNotesFiles in useNotesFilesBackup:
                actNotesFilesBackup = actNotesFiles.copy()
                actNotesFiles = []
                for fileName in actNotesFilesBackup:
                    if ".mp3" in fileName:
                        if not fileName.replace(".mp3", ".wav") in actNotesFilesBackup:
                            # Convert the mp3 to wav
                            path = settings.notesPath + "/" + instrumentsUsed[i] + "/" + fileName
                            #  print("Converting", fileName, "to wav")
                            subprocess.call(["ffmpeg", "-loglevel", "error", "-n",
                                    "-i", path, path.replace(".mp3", ".wav")])
                        actNotesFiles.append(fileName.replace(".mp3", ".wav"))
                    else:
                        actNotesFiles.append(fileName)
                useNotesFiles.append(actNotesFiles)

            useNotesFiles = chooseNotesFiles(useNotesFiles)
            for fileName in useNotesFiles:
                useNotesPaths.append(settings.notesPath + "/" + instrumentsUsed[i] + "/" + fileName)

        if len(useNotesPaths) == 0:
            print("No note files available. Continuing with the next chord.")
            continue

        # Read the files and combine them into one
        signals = []
        sampleRates = []
        for path in useNotesPaths:
            #  print("Reading file:", path)
            sampleRate, signal = wavfile.read(path)
            signalMin = np.ndarray.min(np.absolute(np.array(signal)))
            signalMax = np.ndarray.max(np.absolute(np.array(signal)))
            if signalMin == 0 and signalMax == 0:
                print("One of the files is totally quiet. Continuing with the next chord.")
                err = True
                break
                
            signals.append(signal)
            sampleRates.append(sampleRate)

        if err:
            continue

        # Check if all sampleRates match
        tmp = sampleRates[0]
        for sr in sampleRates:
            if tmp != sr:
                print("Sample rates of note files don't match. Continuing with the next chord.")
                err = True

        if err:
            continue

        minLen = len(signals[0])
        for signal in signals:
            if len(signal) < minLen:
                minLen = len(signal)

        # Trim the signals by the minimum length of the files
        signalsBackup = signals.copy()
        signals = []
        for signal in signalsBackup:
            signals.append(signal[: int(minLen)])

        # Sum the signals
        signal = np.array(sum(signals))

        # Also make it exactly in a range of a short int
        multiplier = (32767 / 2) / np.ndarray.max(np.absolute(signal))
        signal = np.multiply(signal, multiplier).astype(np.int16)

        # TODO Trim the signal in case it gets quiet after some time

        # Figure out a name for the file 
        # (they should be named chord_C_0.wav, chord_C_1.wav, chord_C#m_0.wav, ...)
        fileName = "chord_" + chord + "_"

        # Create settings.chordsPath dir if it does not exist
        subprocess.call(["mkdir", "-p", settings.chordsPath])

        # If there is no file with that name, the number following should be
        # zero. Otherwise, just add 1
        existingFiles = misc.ls(settings.chordsPath)
        existingFiles = [f for f in existingFiles if fileName in f]
        if len(existingFiles) == 0:
            fileName += "0"
        else:
            nums = [f.split(".")[0].split("_")[-1] for f in existingFiles]
            highestNum = max([int(num) for num in nums])
            fileName += str(highestNum + 1)

        # Write the signal to a file
        wavfile.write(settings.chordsPath + "/" + fileName + ".wav", sampleRate, signal)
        print("Chord generated to", settings.chordsPath + "/" + fileName + ".wav")

if __name__ == "__main__":
    main()
