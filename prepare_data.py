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

def saveData(newInputs, newOutputs, fileName, fileNumber):
    print("Saving the data")

    while len(newInputs) != 0:

        # Split the files to max 150k long chunks
        if len(newInputs) > 150000:
            # Save the data (compressed)
            np.savez_compressed(
                    settings.dataDir + "/" + fileName, 
                    inputs = newInputs[0: 150000], outputs = newOutputs[0: 150000])

            newInputs = newInputs[150000: len(newInputs)]
            newOutputs = newOutputs[150000: len(newOutputs)]

            fileName = fileName.replace(str(fileNumber), str(fileNumber + 1))
            fileNumber += 1

        else:
            # Save the data (compressed)
            np.savez_compressed(
                    settings.dataDir + "/" + fileName, 
                    inputs = newInputs, outputs = newOutputs)

            newInputs = []
            newOutputs = []
    print("Data saved")


def getNewFileName(dataFileName):
    fileNumbers = []
    for f in misc.ls(settings.dataDir):
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

def concatOldData(fileName, nnInputs, nnOutputs):
    if fileName in misc.ls(settings.dataDir):
        # Load the old training data (the last file)
        print("Loading the old data")
        oldData = np.load(settings.dataDir + "/" + fileName, allow_pickle=True)
        oldInputs = oldData["inputs"]
        newInputs = np.concatenate((oldInputs, nnInputs))
        oldOutputs = oldData["outputs"]
        newOutputs = np.concatenate((oldOutputs, nnOutputs))
        # Backup the old training data
        print("Creating a backup")
        subprocess.call(["mv",
            settings.dataDir + "/" + fileName,
            settings.dataDir + "/" + fileName.split(".")[0] + "_backup" + "." + fileName.split(".")[1]])

    # Otherwise, just write the nnInputs and nnOutputs
    else:
        newInputs = nnInputs
        newOutputs = nnOutputs

    return newInputs, newOutputs

def processFiles():
    # Arrays where the acquired data will be written
    nnInputs = []
    nnOutputs = []

    # All files to process
    inputFiles = misc.ls(settings.chordsPath)

    for inputFile in inputFiles:
        print("Processing file", inputFiles.index(inputFile) + 1, "of", len(inputFiles))

        # Get the data and save it to the nnArrays
        path = settings.chordsPath + "/" + inputFile
        inputs, outputs, sampleRate = process_audio.processAudio(
                path=path, training=True, verbose=False)
        if len(nnInputs) == 0 and len(nnOutputs) == 0:
            nnInputs = np.array(inputs)
            nnOutputs = np.array(outputs)
        else:
            nnInputs = np.concatenate((nnInputs, inputs))
            nnOutputs = np.concatenate((nnOutputs, outputs))

    print("Done generating inputs and outputs")

    return nnInputs, nnOutputs

def parseArgs():
    if len(sys.argv) < 2 or (sys.argv[1] != "test" and sys.argv[1] != "train"):
        print("Please select the target: test / train")
        quit(0)

    if sys.argv[1] == "test":
        dataFileName = settings.testingDataFileName
    elif sys.argv[1] == "train":
        dataFileName = settings.trainingDataFileName

    return dataFileName

##########
## MAIN ##
##########

def main():
    # The first argument: test / train (target - testing or training data)
    dataFileName = parseArgs()

    # Process all the files (get inputs and outputs)
    nnInputs, nnOutputs = processFiles()

    # Create data dir if it does not exist
    subprocess.call(["mkdir", "-p", settings.dataDir])

    # Get the name of the last training data file
    fileName, fileNumber = getNewFileName(dataFileName)

    # Concat the old data with the new ones if old ones exist
    newInputs, newOutputs = concatOldData(fileName, nnInputs, nnOutputs)

    # Save all the data
    saveData(newInputs, newOutputs, fileName, fileNumber)

    print("Done")

if __name__ == "__main__":
    main()
