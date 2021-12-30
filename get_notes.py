from os import environ
# Disable the GPU because of tensorflow
#  environ["CUDA_VISIBLE_DEVICES"] = "-1"
environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from tensorflow import keras, config
from math import ceil
import sys
import numpy as np

import process_audio
import settings
import misc

###############
## FUNCTIONS ##
###############

def init():
    model = misc.newModel(settings.noteDetNnNodes,
            settings.noteDetHiddenLayersActivationFn,
            settings.noteDetOutputLayerActivationFn,
            settings.noteDetOptimizer,
            settings.noteDetLossFunction
            )
    misc.saveModel(model, settings.noteDetModelPath)
    print("New model created and saved. Please run without init now.")
    quit(0)

def train(model):
    print("Training started, model saved")

    # Load the training data from trainingDataPath (defined in settings.py)
    nnInputs, nnOutputs = misc.loadData(settings.noteDetDataDir, settings.trainingDataFileName)

    # Shuffle the data
    nnInputs, nnOutputs = misc.shuffleData(nnInputs, nnOutputs)

    # Train the neural network (divide the inputs and outputs to
    # inputsPerTraining long arrays) 
    print("Starting training with", len(nnInputs), "data samples")
    for i in range(ceil(len(nnInputs) / settings.noteDetInputsPerTraining)):
        fromIndex = i * settings.noteDetInputsPerTraining
        toIndex = (i + 1) * settings.noteDetInputsPerTraining
        if toIndex > len(nnInputs):
            toIndex = len(nnInputs)
        print("Training with data from index", fromIndex, "to", toIndex)

        # Train the model
        model.fit(nnInputs[fromIndex: toIndex], nnOutputs[fromIndex: toIndex], 
                batch_size=settings.noteDetBatchSize, 
                epochs=settings.noteDetTrainingEpochs, 
                shuffle=True)

    # Save the model
    misc.saveModel(model, settings.noteDetModelPath)
    print("Training finished, model saved")

def test(model):
    print("Testing started")

    # Load the training data from trainingDataPath (defined in settings.py)
    nnInputs, nnOutputs = misc.loadData(settings.noteDetDataDir, settings.testingDataFileName)

    # Shuffle the data
    nnInputs, nnOutputs = misc.shuffleData(nnInputs, nnOutputs)

    print("Starting testing with", len(nnInputs), "data samples")
    predictions = model.predict(nnInputs)
    right = 0
    wrong = 0
    for i in range(len(predictions)):
        expectedChordIndices = np.where(nnOutputs[i] == 1)[0]

        # We can either pick all indices above a certain threshold, which is
        # the more natural way
        threshold = 0.5
        predictedChordIndices = np.where(predictions[i] >= threshold)[0]
        # Or indices of n biggest numbers (n is length of expected indices)
        #  predictedChordIndices = (-predictions[i]).argsort()[: len(expectedChordIndices)]

        for expected in expectedChordIndices:
            if expected in predictedChordIndices:
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

    # Pass the noteMags to the neural network to get the chord for each frame
    predictions = model.predict(nnInputs)

    # Print the predicted chords
    #  print("Predictions: ")
    #  lastIndex = -1
    #  for i in range(len(predictions)):
        #  confidence = max(predictions[i])
        #  chordIndex = np.where(predictions[i] == confidence)[0][0]
        #  if chordIndex != lastIndex:
            #  fromTime = misc.sampleToTime(i * settings.fftStep, sampleRate)
            #  print("From ", "{:.2f}".format(fromTime) + "s:",
                #  "chord ", chordsStrings[chordIndex].ljust(7), 
                #  " with confidence of ", str(int(confidence * 100)) + "%")
            #  lastIndex = chordIndex

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

        config.threading.set_intra_op_parallelism_threads(settings.threadLimit)
        config.threading.set_inter_op_parallelism_threads(settings.threadLimit)

        # Load the neural network (tf model)
        print("Loading the model")
        try:
            print(settings.noteDetModelPath)
            model = keras.models.load_model(settings.noteDetModelPath)
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
