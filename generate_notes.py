from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
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

#  def printUsage():
    #  print("Usage: ")
    #  print("python3 gen_training_data.sh amount instrumentsAmount [targets]")
    #  print("")
    #  print("Amount stands for:")
    #  print("  The amount of .wav samples (chords) to generate")
    #  print("InstrumentsAmount stands for:")
    #  print("  The amount of instruments to use when generating the chord")
    #  print("  (Each instrument plays the chord separately, but audio is merged)")
    #  print("Targets can be (separated by spaces):")
    #  print("  major, minor, 7, 5, dim, dim7, aug, sus2, sus4, maj7, m7, 7sus4")
    #  print("")
    #  print("Notes are expected in ./audio/notes/instrument/")
    #  print("  While instrument can be any string")
    #  print("  Example of a note file: C#0_x.wav, while x goes from 0")
    #  print("Chords will be saved to ./audio/chords/instrument/")
    #  print("  Example of a chord file: C7_x.wav, while x goes from 0")

#  def handleArguments():
    #  args = sys.argv[1: ]

    #  if len(args) < 2:
        #  print("Please specify the amount of chords and amount of instruments")
        #  printUsage()
        #  quit(1)

    #  if args[0] == "--help" or args[0] == "-h":
        #  printUsage()
        #  quit(0)

    #  amount = int(args[0])
    #  instrumentsAmount = int(args[1])

    #  possibleTargets = ["major", "minor", "7", "5", "dim", "dim7", "aug", "sus2", "sus4", "maj7", "m7", "7sus4"]
    #  if len(args) > 2:
        #  targets = args[2: ]
        #  for target in targets:
            #  if not target in possibleTargets:
                #  print("Wrong target:", target)
                #  printUsage()
                #  quit(1)
    #  else:
        #  targets = possibleTargets

    #  return amount, instrumentsAmount, targets

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


##########
## MAIN ##
##########

def main():

    print("==================================================")
    print("==================================================")

    #  amount, instrumentsAmount, targets = handleArguments()
    try:
        amount = int(sys.argv[1])
        instrumentsAmount = int(sys.argv[2])
    except:
        print("Wrong arguments")
        quit(0)

    # Get all instruments (folders in settings.notesPath)
    instruments = misc.ls(settings.notesPath)
    if "0_short" in instruments:
        instruments.remove("0_short")

    # For amount
    for i in range(amount):
        err = False

        print("==================================================")

        instrumentsUsed = []
        orders = []
        for i in range(instrumentsAmount):
            # Randomly pick an instrument
            instrumentsUsed.append(instruments[np.random.randint(len(instruments))])
            if not settings.noteGenMixInstruments:
                instrumentsUsed[i] = instrumentsUsed[0]

            # Randomly pick an order
            orders.append(np.random.randint(settings.minOrder, settings.maxOrder + 1))

        print("Instruments picked: ", instrumentsUsed)

        useNotesPaths = []
        for i in range(instrumentsAmount):

            # Get all notes for each instrument (.wav files in settings.notesPath/instrument)
            useNotesFiles = misc.ls(settings.notesPath + "/" + instrumentsUsed[i])

            # Filter out those with orders allowed in settings
            for f in useNotesFiles.copy():
                order = [int(c) for c in f if c.isdigit()][0]
                if order < settings.minNoteOrder or order > settings.maxNoteOrder:
                    useNotesFiles.remove(f)

            # If it is a mp3 file, check if there is a wav with the same name. If
            # not, convert it to a wav file with the same name
            useNotesFilesBackup = useNotesFiles.copy()
            useNotesFiles = []
            for fileName in useNotesFilesBackup:
                if ".mp3" in fileName:
                    if not fileName.replace(".mp3", ".wav") in actNotesFilesBackup:
                        # Convert the mp3 to wav
                        path = settings.notesPath + "/" + instrumentsUsed[i] + "/" + fileName
                        #  print("Converting", fileName, "to wav")
                        subprocess.call(["ffmpeg", "-loglevel", "error", "-n",
                                "-i", path, path.replace(".mp3", ".wav")])
                    useNotesFile.append(fileName.replace(".mp3", ".wav"))
                else:
                    useNotesFiles.append(fileName)

            fileName = useNotesFiles[np.random.randint(len(useNotesFiles))]

            useNotesPaths.append(settings.notesPath + "/" + instrumentsUsed[i] + "/" + fileName)

        if len(useNotesPaths) == 0:
            print("No note files available. Continuing.")
            continue

        print("Files used:")
        for path in useNotesPaths:
            print(path)

        # Read the files and combine them into one
        signals = []
        sampleRates = []
        for path in useNotesPaths:
            #  print("Reading file:", path)
            sampleRate, signal = wavfile.read(path)

            # Crop the signal by loudness (remove the quiet part)
            croppedSignal = []
            for sig in np.transpose(signal):
                sig = misc.cropByLoudness(sig, verbose = False)

                if len(sig) < settings.fftWidth:
                    print("One of the files is totally quiet. Continuing.")
                    err = True
                    break

                signals.append(sig)

            sampleRates.append(sampleRate)

        if err:
            continue

        # Check if all sampleRates match
        tmp = sampleRates[0]
        for sr in sampleRates:
            if tmp != sr:
                print("Sample rates of note files don't match. Continuing.")
                err = True

        if err:
            continue

        minLen = len(signals[0])
        for signal in signals:
            if len(signal) < minLen:
                minLen = len(signal)
        print("New file length:", "{:.2f}".format(minLen / sampleRate), "seconds")

        # Trim the signals by the minimum length of the files
        signalsBackup = signals.copy()
        signals = []
        for signal in signalsBackup:
            signals.append(signal[: int(minLen)])

        # Sum the signals
        signal = np.array(sum(signals))

        # Also make it exactly in a range of a short int
        multiplier = (32767 / 2) / max(abs(signal))
        signal = np.multiply(signal, multiplier).astype(np.int16)

        # Create dir if it does not exist
        subprocess.call(["mkdir", "-p", settings.generatedNotesPath])

        # Figure out a name for the file 
        fileName = "note_"
        for path in useNotesPaths:
            note = path.split("/")[-1].split("_")[0]
            fileName += note + "_"

        # If there is no file with that name, the number following should be
        # zero. Otherwise, just add 1
        existingFiles = misc.ls(settings.generatedNotesPath)
        existingFiles = [f for f in existingFiles if fileName in f]
        if len(existingFiles) == 0:
            fileName += "0"
        else:
            nums = [f.split(".")[0].split("_")[-1] for f in existingFiles]
            highestNum = max([int(num) for num in nums])
            fileName += str(highestNum + 1)

        # Write the signal to a file
        wavfile.write(settings.generatedNotesPath + "/" + fileName + ".wav", sampleRate, signal)
        print("Notes generated to", settings.generatedNotesPath + "/" + fileName + ".wav")

if __name__ == "__main__":
    main()
