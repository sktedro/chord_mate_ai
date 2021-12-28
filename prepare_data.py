import sys
import numpy as np
import subprocess
import process_audio
import settings

# Notes: preparing data always appends to the old data, does not overwrite

###############
## FUNCTIONS ##
###############

def ls(path):
    return subprocess.run(["ls", path], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]

##########
## MAIN ##
##########

def main():
    # TODO ARGUMENTS PARSING
    # test / train (target - testing or training data)
    if len(sys.argv) < 2 or (sys.argv[1] != "test" and sys.argv[1] != "train"):
        print("Please select the target: test / train")
        quit(0)

    if sys.argv[1] == "test":
        dataFileName = settings.testingDataFileName
    elif sys.argv[1] == "train":
        dataFileName = settings.trainingDataFileName

    # Arrays where the acquired data will be written
    nnInputs = []
    nnOutputs = []

    inputFiles = ls(settings.chordsPath)

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

    # Create data dir if it does not exist
    subprocess.call(["mkdir", "-p", settings.dataDir])

    # Concat the old data with the new ones if old ones exist
    if len(ls(settings.dataDir)) and dataFileName in ls(settings.dataDir):
        # Load the old training data
        oldData = np.load(settings.dataDir + "/" + dataFileName, allow_pickle=True)
        oldInputs = oldData["inputs"]
        newInputs = np.concatenate((oldInputs, nnInputs))
        oldOutputs = oldData["outputs"]
        newOutputs = np.concatenate((oldOutputs, nnOutputs))
        # Backup the old training data
        subprocess.call(["mv", 
        settings.dataDir + "/" + dataFileName, 
        settings.dataDir + "/" + dataFileName + "_backup"])

    # Otherwise, just write the nnInputs and nnOutputs
    else:
        newInputs = nnInputs
        newOutputs = nnOutputs

    print("Saving the data")

    # Output the data (compressed)
    np.savez_compressed(
            settings.dataDir + "/" + dataFileName, 
            inputs = newInputs, outputs = newOutputs)

    print("Done")

if __name__ == "__main__":
    main()
