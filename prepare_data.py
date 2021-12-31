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
        audioDir = settings.chordsPath
        processFunction = process_audio.processAudio
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
