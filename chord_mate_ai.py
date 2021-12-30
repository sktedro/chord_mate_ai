from os import environ
# Disable the GPU because of tensorflow
#  environ["CUDA_VISIBLE_DEVICES"] = "-1"
environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from tensorflow import keras
import sys
import numpy as np

import process_audio
import settings
from math import ceil
import misc

###############
## FUNCTIONS ##
###############

def init():
    model = misc.newModel(settings.nnNodes,
            settings.hiddenLayersActivationFn,
            settings.outputLayerActivationFn,
            settings.optimizer,
            settings.lossFunction
            )
    misc.saveModel(model, settings.modelPath)
    print("New model created and saved. Please run without init now.")
    quit(0)

def train(model):
    print("Training started, model saved")

    # Load the training data from trainingDataPath (defined in settings.py)
    nnInputs, nnOutputs = misc.loadData(settings.dataDir, settings.trainingDataFileName)

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
    misc.saveModel(model, settings.modelPath)
    print("Training finished, model saved")

def test(model):
    print("Testing started")

    # Load the training data from trainingDataPath (defined in settings.py)
    nnInputs, nnOutputs = misc.loadData(settings.dataDir, settings.testingDataFileName)

    # Shuffle the data
    nnInputs, nnOutputs = misc.shuffleData(nnInputs, nnOutputs)

    # Just an array of all 144 chords
    chordsStrings = misc.getChordsStringsArray()

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
        print("Accuracy: ", "{:.4f}".format(accuracy) + "% ", end = "\r")
    print("Accuracy: ", "{:.4f}".format(accuracy) + "% ")

    print("Testing finished")

def predict(model, nnInputs, sampleRate):

    # Just an array of all 144 chords
    chordsStrings = misc.getChordsStringsArray()

    # Pass the noteMags to the neural network to get the chord for each frame
    predictions = model.predict(nnInputs)

    # Print the predicted chords
    print("Predictions: ")
    lastIndex = -1
    for i in range(len(predictions)):
        confidence = max(predictions[i])
        chordIndex = np.where(predictions[i] == confidence)[0][0]
        if chordIndex != lastIndex:
            fromTime = misc.sampleToTime(i * settings.fftStep, sampleRate)
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
            fromTime = misc.sampleToTime(i * settings.fftStep, sampleRate)
            print("From ", "{:.2f}".format(fromTime) + "s:",
                "chord ", chordsStrings[chordIndex].ljust(7))
            lastIndex = chordIndex

##########
## MAIN ##
##########

def main():
    print("==================================================")
    # Whatever the user does, he needs to specify at least one argument
    argsCount = len(sys.argv)
    if argsCount < 2:
        print("ERROR: Please specify an audio file or a special command")
        quit(1)

    # Initialize a new model
    if sys.argv[1] == "init":
        init()

    else:
        # Load the neural network (tf model)
        print("Loading the model")
        try:
            model = keras.models.load_model(settings.modelPath)
        except:
            print("ERROR: Model not found")
            quit(1)

        # Train
        if sys.argv[1] == "train":
            train(model)

        # Test
        elif sys.argv[1] == "test":
            test(model)

        # Predict
        else:
            nnInputs, nnOutputs, sampleRate = process_audio.processAudio(
                    path=sys.argv[1], training=False, verbose=True)
            predict(model, nnInputs, sampleRate)

###############
## MAIN CALL ##
###############

if __name__ == "__main__":
    main();
