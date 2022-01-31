import sys
import numpy as np
import subprocess
import process_audio
import settings
import misc

# Notes: preparing data always appends to the old data, does not overwrite

###############
## FUNCTIONS ##
###############

def processNotes(path, training, verbose):
    # Get the file name and convert to wav if needed
    fileName = getFileName(path)

    # Read the wav file
    sampleRate, samples = wavfile.read(path)
    if sampleRate != 44100:
        print("Sorry, no other sampling frequency than 44100Hz is currently supported")
        quit(0)

    # Perform the fourier transform
    freqs, magnitudes = process_audio.fourier_transform(samples, sampleRate, verbose)

    # Crop the magnitudes to only the frequency bins that are useful to us
    notes, noteFreqs = misc.getNoteFreqs()
    freqsToPass = []
    fftResolution = 12
    for noteFreq in noteFreqs:
        freqBin = np.arange(noteFreq - fftResolution / 2, noteFreq + fftResolution / 2, 1)
        for f in freqBin:
            freqsToPass.append(f)
    freqsToPass = np.unique(np.round(freqsToPass).astype(np.int16))

    nnInputs = []
    for mags in magnitudes:
        tmpInputs = []
        for f in freqsToPass:
            tmpInputs.append(mags[f])
        nnInputs.append(tmpInputs)


    nnOutputs = []
    if training:
        # Get note strings in an array shaped the same as predictions
        notesStringsSharp = misc.getNotesStringsArray(semitoneChar="#")
        notesStringsFlat = misc.getNotesStringsArray(semitoneChar="b")

        notes = fileName.split("_")[1: -1]
        output = np.zeros(12)
        for note in notes:
            if note in notesStringsSharp:
                index = np.where(notesStringsSharp == note)
            elif note in notesStringsFlat:
                index = np.where(notesStringsFlat == note)
            else:
                print("Invalid file name:", fileName)
                quit(1)
            if len(index) > 0:
                index = index[0]
            output[index % 12] = 1.0
        print("Notes contained:",
                [str(notesStringsSharp[i]).replace("0", "")
                    for i in range(len(output)) if output[i] == 1])

        for i in range(len(nnInputs)):
            nnOutputs.append(output)

    nnInputs = np.array(nnInputs)
    nnOutputs = np.array(nnOutputs)
    #  print(nnInputs.shape)
    #  print(nnOutputs.shape)
    #  print(nnInputs)
    #  print(nnOutputs)

    return nnInputs, nnOutputs, sampleRate


def processChords(path, training, verbose):
    # Get the file name and convert to wav if needed
    fileName = getFileName(path)

    # Read the wav file
    sampleRate, samples = wavfile.read(path)
    if sampleRate != 44100:
        print("Sorry, no other sampling frequency than 44100Hz is currently supported")
        quit(0)

    # Perform the fourier transform
    freqs, magnitudes = process_audio.fourier_transform(samples, sampleRate, verbose)

    # Get notes, exact notes freqs and magnitudes
    notes, noteFreqs, noteMags = getNoteMagnitudes(magnitudes, freqs)

    # Arrays where the acquired data will be written
    nnInputs = noteMags

    nnOutputs = []
    if training:
        # Get chords strings in an array shaped the same as predictions
        chordsStrings = misc.getChordsStringsArray()

        chordIndex = chordsStrings.index(fileName.split("_")[1])
        output = np.zeros(144)
        output[chordIndex] = 1.0
        for i in range(len(noteMags)):
            nnOutputs.append(output)

    nnInputs = np.array(nnInputs)
    nnOutputs = np.array(nnOutputs)

    return nnInputs, nnOutputs, sampleRate

def saveData(dataDir, newInputs, newOutputs, fileName, fileNumber, fileDataLimit):
    print("Saving the data")

    while len(newInputs) != 0:

        # Split the files to max 150k long chunks
        if len(newInputs) > fileDataLimit:
            # Save the data (compressed)
            np.savez_compressed(
                    dataDir + "/" + fileName, 
                    inputs = newInputs[0: fileDataLimit], outputs = newOutputs[0: fileDataLimit])

            newInputs = newInputs[fileDataLimit: len(newInputs)]
            newOutputs = newOutputs[fileDataLimit: len(newOutputs)]

            fileName = fileName.replace(str(fileNumber), str(fileNumber + 1))
            fileNumber += 1

        else:
            # Save the data (compressed)
            np.savez_compressed(
                    dataDir + "/" + fileName, 
                    inputs = newInputs, outputs = newOutputs)

            newInputs = []
            newOutputs = []
    print("Data saved")

def getNewFileName(dataDir, dataFileName):
    fileNumbers = []
    for f in misc.ls(dataDir):
        if (not dataFileName in f) or "backup" in f:
            continue
        fileNumbers.append(f.split(".")[0].split("_")[-1])
    fileNumber = [int(i) for i in fileNumbers]
    fileNumber.sort()
    if len(fileNumber) > 0:
        fileNumber = fileNumber[-1]
    else:
        fileNumber = 0
    fileName = dataFileName + "_" + str(fileNumber) + ".npz"
    return fileName, fileNumber

def concatOldData(dataDir, fileName, nnInputs, nnOutputs):
    if fileName in misc.ls(dataDir):
        # Load the old training data (the last file)
        print("Loading the old data")
        oldData = np.load(dataDir + "/" + fileName, allow_pickle=True)
        oldInputs = oldData["inputs"]
        newInputs = np.concatenate((oldInputs, nnInputs))
        oldOutputs = oldData["outputs"]
        newOutputs = np.concatenate((oldOutputs, nnOutputs))
        # Backup the old training data
        print("Creating a backup")
        subprocess.call(["mv",
            dataDir + "/" + fileName,
            dataDir + "/" + fileName.split(".")[0] + "_backup" + "." + fileName.split(".")[1]])

    # Otherwise, just write the nnInputs and nnOutputs
    else:
        newInputs = nnInputs
        newOutputs = nnOutputs

    return newInputs, newOutputs

def processFiles(audioDir, processFunction):
    # Arrays where the acquired data will be written
    nnInputs = []
    nnOutputs = []

    # All files to process
    inputFiles = misc.ls(audioDir)

    for inputFile in inputFiles:
        print("Processing file", inputFiles.index(inputFile) + 1, "of", len(inputFiles))

        # Get the data and save it to the nnArrays
        path = audioDir + "/" + inputFile
        inputs, outputs, sampleRate = processFunction(
                path=path, training=True, verbose=False)
        inputs = np.transpose(np.transpose(inputs)[: settings.fs // 4])

        if len(nnInputs) == 0 and len(nnOutputs) == 0:
            nnInputs = np.array(inputs)
            nnOutputs = np.array(outputs)
        else:
            nnInputs = np.concatenate((nnInputs, inputs))
            nnOutputs = np.concatenate((nnOutputs, outputs))

    print("Done generating inputs and outputs")

    return nnInputs, nnOutputs

def parseArgs():
    if len(sys.argv) < 3:
        print("Please select the targets: test / train and notes / chords")
        quit(0)

    if sys.argv[1] == "test":
        dataFileName = settings.testingDataFileName
    elif sys.argv[1] == "train":
        dataFileName = settings.trainingDataFileName
    else:
        print("Unknown target:", sys.argv[1])
        quit(0)

    # 410kB per wav file with 816 inputs and 12 outputs
    if sys.argv[2] == "notes":
        dataDir = settings.noteDetDataDir
        audioDir = settings.generatedNotesPath
        processFunction = process_audio.processNotes
        fileDataLimit = 10000
    elif sys.argv[2] == "chords":
        dataDir = settings.dataDir
        audioDir = settings.generatedChordsPath
        processFunction = process_audio.processChords
        fileDataLimit = 150000 # TODO
    else:
        print("Unknown target:", sys.argv[2])
        quit(0)

    return audioDir, dataDir, dataFileName, processFunction, fileDataLimit

##########
## MAIN ##
##########

def main():
    # The first argument: test / train (target - testing or training data)
    audioDir, dataDir, dataFileName, processFunction, fileDataLimit = parseArgs()

    # Process all the files (get inputs and outputs)
    nnInputs, nnOutputs = processFiles(audioDir, processFunction)

    # Create data dir if it does not exist
    subprocess.call(["mkdir", "-p", dataDir])

    # Get the name of the last training data file
    fileName, fileNumber = getNewFileName(dataDir, dataFileName)

    # Concat the old data with the new ones if old ones exist
    newInputs, newOutputs = concatOldData(dataDir, fileName, nnInputs, nnOutputs)

    # Save all the data
    saveData(dataDir, newInputs, newOutputs, fileName, fileNumber, fileDataLimit)

    print("Done")

if __name__ == "__main__":
    main()
