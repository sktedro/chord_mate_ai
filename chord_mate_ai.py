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
import misc

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
    # Create a sequential model
    model = tf.keras.Sequential()

    # Input layer
    model.add(
            tf.keras.Input(
                shape=(settings.nnNodes[0], )))

    # Hidden layers
    for nodes in settings.nnNodes[1: -1]:
        model.add(
                tf.keras.layers.Dense(
                    nodes,
                    activation=settings.hiddenLayersActivationFn))
    # Output layer
    model.add(
            tf.keras.layers.Dense(
                settings.nnNodes[-1],
                activation=settings.outputLayerActivationFn))

    # Compiling the model
    model.compile(
            optimizer=settings.optimizer,
            loss=settings.lossFunction,
            metrics=["accuracy"])

    # To complete the model, a prediction must be made for whatever reason
    randomInput = np.array([np.random.random(settings.nnNodes[0])])
    randomPredict = model.predict(randomInput)

    return model

def saveModel(model):
    model.save(settings.modelPath)

def train(model):
    # Load the training data from trainingDataPath (defined in settings.py)
    nnInputs, nnOutputs = misc.loadData(settings.trainingDataFileName)

    # Shuffle the data
    nnInputs, nnOutputs = misc.shuffleData(nnInputs, nnOutputs)

    # Train the neural network (divide the inputs and outputs to
    # inputsPerTraining long arrays) 
    print("Starting training with", len(nnInputs), "data samples")
    for i in range(ceil(len(nnInputs) / settings.inputsPerTraining)):
        fromIndex = i * settings.inputsPerTraining
        toIndex = (i + 1) * settings.inputsPerTraining
        if toIndex > len(nnInputs):
            toIndex = len(nnInputs)
        print("Training with data from index", fromIndex, "to", toIndex)

        # Train the model
        model.fit(nnInputs[fromIndex: toIndex], nnOutputs[fromIndex: toIndex], 
                batch_size=settings.batchSize, epochs=settings.trainingEpochs, 
                shuffle=True)

    # Save the model
    saveModel(model)
    print("Training finished, model saved")

def test(model):
    # Load the training data from trainingDataPath (defined in settings.py)
    nnInputs, nnOutputs = misc.loadData(settings.testingDataFileName)

    # Just an array of all 144 chords
    chordsStrings = process_audio.getChordsStringsArray()

    print("Starting testing with", len(nnInputs), "data samples")
    predictions = model.predict(nnInputs)
    right = 0
    wrong = 0
    for i in range(len(predictions)):
        expectedChordIndex = np.where(nnOutputs[i] == 1)[0][0]
        predictedChordIndex = np.where(predictions[i] == max(predictions[i]))[0][0]
        expected = chordsStrings[expectedChordIndex]
        predicted = chordsStrings[predictedChordIndex]
        if expected == predicted:
            right += 1
        else:
            wrong += 1
        accuracy = 100 * right / (right + wrong)
        #  print("Expected", expected.ljust(7), ", predicted", predicted.ljust(7),
                #  " | Accuracy: ", "{:.4f}".format(accuracy) + "%")
        print("Accuracy: ", "{:.4f}".format(accuracy) + "%", end = "\r")
    print("Accuracy: ", "{:.4f}".format(accuracy) + "%")

    print("Testing finished")

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
        elif sys.argv[1] == "test":
            test(model)

        else:
            nnInputs, nnOutputs, sampleRate = process_audio.processAudio(
                    path=sys.argv[1], training=False, verbose=True)
            predict(model, nnInputs, sampleRate)

###############
## MAIN CALL ##
###############

if __name__ == "__main__":
    main();
