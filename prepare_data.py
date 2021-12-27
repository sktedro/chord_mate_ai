import sys
import numpy as np
import subprocess
import process_audio
import settings

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
    # -s --shuffle to just shuffle
    # -a --append to append to the existing file (will overwrite without it)

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
    subprocess.call(["mkdir", "-p", settings.trainingDataDir])

    # Concat the old data with the new ones if old ones exist
    if len(ls(settings.trainingDataDir)) and ls(settings.trainingDataDir)[0] == settings.trainingDataFileName:
        oldData = np.load(settings.trainingDataDir + "/" + settings.trainingDataFileName, allow_pickle=True)
        oldInputs = oldData["inputs"]
        newInputs = np.concatenate((oldInputs, nnInputs))
        oldOutputs = oldData["outputs"]
        newOutputs = np.concatenate((oldOutputs, nnOutputs))
    # Otherwise, just write the nnInputs and nnOutputs
    else:
        newInputs = nnInputs
        newOutputs = nnOutputs

    # Output the data (compressed)
    print("Saving the data")
    np.savez_compressed(
            settings.trainingDataDir + "/" + settings.trainingDataFileName, 
            inputs = newInputs, outputs = newOutputs)
    print("Done")

if __name__ == "__main__":
    main()
