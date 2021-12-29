import numpy as np
import subprocess
import settings

def ls(path):
    return subprocess.run(["ls", path], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]

def shuffleData(inputs, outputs):
    print("Shuffling the data")

    # Create random indices
    destIndices = np.arange(len(inputs))
    np.random.shuffle(destIndices)
    inputsBackup = inputs.copy()
    outputsBackup = outputs.copy()

    # Move the inputs and outputs to a new array based on these indices
    for i in range(len(destIndices)):
        inputs[i] = inputsBackup[destIndices[i]]
        outputs[i] = outputsBackup[destIndices[i]]
    return inputs, outputs

def loadData(fileName):
    print("Loading the data")
    inputs = []
    outputs = []
    for f in ls(settings.dataDir):
        if (not fileName in f) or "backup" in f:
            continue
        print(f)
        data = np.load(settings.dataDir + "/" + f, allow_pickle=True)
        if len(inputs) == 0 and len(outputs) == 0:
            inputs = data["inputs"]
            outputs = data["outputs"]
        else:
            inputs = np.concatenate((inputs, data["inputs"]))
            outputs = np.concatenate((outputs, data["outputs"]))

    if len(inputs) == 0 and len(outputs) == 0:
        print("No data available. Please run 'prepare_data' first.")
        quit(0)

    return inputs, outputs


