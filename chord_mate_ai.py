import os
# Disable the GPU because of tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from math import e
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wave
import subprocess
import prepare_data
import process_audio
import settings
from math import ceil

###############
## FUNCTIONS ##
###############

def sampleToTime(sampleNum, sampleRate):
    return sampleNum / sampleRate

# Layers:
# Input layer: 
    # Inputs: Magnitudes of tones (A0, A#0, ..., G#0, A1, ..., G#7)
    # Inputs amount: 12 * 8 = 96
# Output layer:
    # Outputs: Chords (A, A#, ..., G)(major, minor, 7, 5, ...)
    # Outputs amount: 12 * 12 = 144
# TODO Also input some history or metadata?
def newModel():
    inputNodes = 96
    hiddenLayer1Nodes = 512
    hiddenLayer2Nodes = 256
    outputNodes = 144

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(settings.nnNodes[0], )))
    for nodes in settings.nnNodes[1: -1]:
        model.add(tf.keras.layers.Dense(nodes, activation="tanh"))
    model.add(tf.keras.layers.Dense(settings.nnNodes[-1], activation="tanh"))

    model.compile(optimizer="adam", loss="mean_squared_error")

    randomInput = np.array([np.random.random(inputNodes)])
    randomPredict = model.predict(randomInput)

    return model

def saveModel(model):
    model.save(settings.modelPath)

def ls(path):
    return subprocess.run(["ls", path], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]

def train(model):
    # Load the training data from trainingDataPath (defined in prepare_data.py)
    print("Loading the data")
    if ls(settings.trainingDataDir)[0] != settings.trainingDataFileName:
        print("No data available. Please run 'prepare_data' first.")
        quit(0)

    # Load the data and shuffle it
    data = np.load(settings.trainingDataDir + "/" + settings.trainingDataFileName, allow_pickle=True)
    nnInputs = data["inputs"]
    nnOutputs = data["outputs"]

    # Shuffle the data
    print("Shuffling the data")
    # Create random indices
    destIndices = np.arange(len(nnInputs))
    np.random.shuffle(destIndices)
    nnInputsBackup = nnInputs
    nnOutputsBackup = nnOutputs
    # Move the inputs and outputs to a new array based on these indices
    for i in range(len(destIndices)):
        nnInputs[i] = nnInputsBackup[destIndices[i]]
        nnOutputs[i] = nnOutputsBackup[destIndices[i]]

    # Train the neural network (divide the inputs and outputs to
    # inputsPerTraining long arrays) 
    print("Starting training with", len(nnInputs), "data samples")
    for i in range(ceil(len(nnInputs) / settings.inputsPerTraining)):
        fromIndex = i * settings.inputsPerTraining
        toIndex = (i + 1) * settings.inputsPerTraining
        if toIndex > len(nnInputs):
            toIndex = len(nnInputs)
        print("From", fromIndex, "to", toIndex)

        # Train the model
        model.fit(nnInputs[fromIndex: toIndex], nnOutputs[fromIndex: toIndex], 
                batch_size=settings.batchSize, epochs=settings.trainingEpochs, 
                shuffle=True)

    # Save the model
    saveModel(model)
    print("Training finished, model saved")

def predict(model, nnInputs, sampleRate):

    # Just an array of all 144 chords
    chordsStrings = process_audio.getChordsStringsArray()

    # Pass the noteMags to the neural network to get the chord for each frame
    predictions = model.predict(nnInputs)

    # Print the predicted chords
    print("Predictions: ")
    lastIndex = -1
    for i in range(len(predictions)):
        confidence = max(predictions[i])
        chordIndex = np.where(predictions[i] == confidence)[0][0]
        if chordIndex != lastIndex:
            fromTime = sampleToTime(i * settings.fftStep, sampleRate)
            print("From ", "{:.2f}".format(fromTime) + "s:",
                "chord ", chordsStrings[chordIndex].ljust(7), 
                " with confidence of ", str(int(confidence * 100)) + "%")
            lastIndex = chordIndex

    lastIndex = -1
    print("After filtering by confidence:")
    for i in range(len(predictions)):
        confidence = max(predictions[i])
        chordIndex = np.where(predictions[i] == confidence)[0][0]
        if chordIndex != lastIndex and confidence > 0.8:
            fromTime = sampleToTime(i * settings.fftStep, sampleRate)
            print("From ", "{:.2f}".format(fromTime) + "s:",
                "chord ", chordsStrings[chordIndex].ljust(7))
            lastIndex = chordIndex

##########
## MAIN ##
##########

# TODO call the generator from here - don't let it generate sample files but
# return the wave here

def main():
    # Whatever the user does, he needs to specify at least one argument
    argsCount = len(sys.argv)
    if argsCount < 2:
        print("ERROR: Please specify an audio file or a special command")
        quit(1)

    # Create and save a new model if the user wants to
    if sys.argv[1] == "init":
        # TODO are you sure?
        model = newModel()
        saveModel(model)
        print("New model created and saved. Please run without init now.")
        quit(0)

    else:
        # Load the neural network (tf model)
        # TODO check if there is a model saved
        print("Loading the model")
        model = tf.keras.models.load_model(settings.modelPath)

        if sys.argv[1] == "train":
            train(model)

        else:
            nnInputs, nnOutputs, sampleRate = process_audio.processAudio(
                    path=sys.argv[1], training=False, verbose=True)
            predict(model, nnInputs, sampleRate)

###############
## MAIN CALL ##
###############

if __name__ == "__main__":
    main();
