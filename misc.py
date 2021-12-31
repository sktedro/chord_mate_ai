import numpy as np
from tensorflow import keras
import subprocess
import settings

def ls(path):
    return subprocess.run(["ls", path], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]

def sampleToTime(sampleNum, sampleRate):
    return sampleNum / sampleRate

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

def loadData(dirName, fileName):
    print("Loading the data")
    inputs = []
    outputs = []
    for f in ls(dirName):
        if (not fileName in f) or "backup" in f:
            continue
        print(f)
        data = np.load(dirName + "/" + f, allow_pickle=True)
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

def getNotesStringsArray(semitoneChar):
    if semitoneChar == "#":
        notes = np.array([
            "C0", "C#0", "D0", "D#0", "E0", "F0",
            "F#0", "G0", "G#0", "A0", "A#0", "B0"])
    elif semitoneChar == "b":
        notes = np.array([
            "C0", "Db0", "D0", "Eb0", "E0", "F0",
            "Gb0", "G0", "Ab0", "A0", "Bb0", "B0"])

    for i in range(7):
        notes = np.concatenate(
                (notes, np.char.replace(notes[0: 12], "0", str(i + 1))))

    return notes

def getChordsStringsArray():
    major = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    minor = [major + "m" for major in major]
    seventh = [major + "7" for major in major]
    fifth = [major + "5" for major in major]
    dim = [major + "dim" for major in major]
    dim7 = [major + "dim7" for major in major]
    aug = [major + "aug" for major in major]
    sus2 = [major + "sus2" for major in major]
    sus4 = [major + "sus4" for major in major]
    maj7 = [major + "maj7" for major in major]
    m7 = [major + "m7" for major in major]
    seventhsus4 = [major + "7sus4" for major in major]
    return major + minor + seventh + fifth + dim + dim7 + aug + sus2 + sus4 + maj7 + m7 + seventhsus4

def getNoteFreqs():
    # TODO Discard the notes of order 0 as the resolution is too low here?
    # Maybe even order 1? Or distribute the magnitudes later?

    notes = np.array([
        "C0", "C#0", "D0", "D#0", "E0", "F0",
        "F#0", "G0", "G#0", "A0", "A#0", "B0"])

    noteFreqs = np.array([
    #       C0     C#0    D0     D#0
            16.35, 17.32, 18.35, 19.45,
    #       E0     F0     F#0    G0
            20.60, 21.83, 23.12, 24.50,
    #       G#0    A0     A#0    B0
            25.96, 27.50, 29.14, 30.87 ])

    # Finish the notes and noteFreqs sequences (we have them for order 0, we
    # need them for orders 0 to 7)
    for i in range(7):
        notes = np.concatenate((notes, np.char.replace(notes[0: 12], "0", str(i + 1))))
        noteFreqs = np.concatenate((noteFreqs, noteFreqs[i * 12: (i + 1) * 12] * 2))

    return notes, noteFreqs


# TODO Also input some history or metadata?
def newModel(nnNodes, hiddenActivationFn, outputActivationFn, opt, lossFn):
    # Create a sequential model
    model = keras.Sequential()

    # Input layer
    model.add(
            keras.Input(
                shape=(nnNodes[0], )))

    # Hidden layers
    for nodes in nnNodes[1: -1]:
        model.add(
                keras.layers.Dense(
                    nodes,
                    activation=hiddenActivationFn))
    # Output layer
    model.add(
            keras.layers.Dense(
                nnNodes[-1],
                activation=outputActivationFn))

    # Compiling the model
    model.compile(
            optimizer=opt,
            loss=lossFn,
            metrics=["accuracy"])

    # To complete the model, a prediction must be made for whatever reason
    randomInput = np.array([np.random.random(nnNodes[0])])
    randomPredict = model.predict(randomInput)

    return model

def saveModel(model, path):
    if path.split("/")[-1] in ls("./"):
        subprocess.run(
                ["rm", "-rf", path + "_backup"], 
                stdout=subprocess.PIPE)
        subprocess.run(
                ["mv", path, path + "_backup"], 
                stdout=subprocess.PIPE)
    model.save(path)


