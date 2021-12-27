import os
# Disable the GPU because of tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
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

##############
## SETTINGS ##
##############

fftWidth = 8192
fftStep = 2048
# With sample rate 44100, this makes the FFT accuracy around 5Hz
# TODO Make it adaptive to the sampling frequency? Or don't accept files with
# Fs other than 44100?

chordsPath = "./audio/chords"
audioPath = "./audio"
modelPath = "./nn/"

###############
## FUNCTIONS ##
###############

def normalizeSamples(samples, sampleCount):
    # Subtract the average
    avgVal = sum(samples) / sampleCount
    normSamples = samples - avgVal
    # Divide by the maximum value
    normSamples = normSamples / max((normSamples))
    return normSamples

def getFrames(samples, sampleCount):
    # Prepare the frames
    frames = []
    for i in range(sampleCount // fftStep - 1):
        frames.append(samples[i * fftStep : i * fftStep + fftWidth])
    #  print("We have", len(frames), "frames, all with length", len(frames[0]))
    return frames

def transformSignal(frames, sampleRate):
    #  print("Calculating the DFT")
    # FFT
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: fftWidth // 2] for k in range(len(frames))]))
    freqs = [k * sampleRate // fftWidth for k in range(fftWidth // 2)]
    #  print("DFT calculated")
    return magnitudes, freqs

def plotDft(freqs, magnitudes):
    plt.plot(freqs, magnitudes)
    plt.title("Discrete fourier transform")
    plt.ylabel("Magnitude []")
    plt.xlabel("Frequency [Hz]")
    plt.show()

def sampleToTime(sampleNum, sampleRate):
    return sampleNum / sampleRate

def plotSignal(samples, duration):
    plt.plot(np.arange(0, duration, duration / len(samples)), samples)
    plt.ylabel("Amplitude []")
    plt.xlabel("Time [s]")
    plt.title("One frame")
    plt.show()


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

    #  print("Notes: ")
    #  print(notes)

    #  print("Notes freqs: ")
    #  for i in range(8):
        #  print(noteFreqs[i * 12: (i + 1) * 12])

    return notes, noteFreqs

# TODO static analysis of this function
# For each frequency in freqs, get indices of note freqs that
# are contained by the frequency
# I know this is hard to understand (I had a slightly tough time while
# coding this, too), sorry
def getNotesContainedInFreqs(noteFreqs, freqs, resolution):
    spread = resolution / 2
    
    notesContained = []
    # https://dsp.stackexchange.com/questions/26927/what-is-a-frequency-bin
    # https://stackoverflow.com/questions/10754549/fft-bin-width-clarification
    # Both these sources say that the fft coeff represents the center
    # of a frequency bin
    lowerBound = np.array(freqs) - spread
    upperBound = np.array(freqs) + spread
    # If the frequency is within resolution/2 of a note, copy the magnitude
    # Watch out if it is within resolution/2 of several notes - in that
    # case, distribute it over all of them
    for i in range(len(freqs)):

        # Save indices of those notes, of which freqs are contained 
        # in the fft sample
        notesContainedTmp = np.zeros([0])
        startIndex = np.where(noteFreqs >= lowerBound[i])[0]
        endIndex = np.where(noteFreqs <= upperBound[i])[0]
        if len(startIndex) and len(endIndex):
            startIndex = startIndex[0]
            endIndex = endIndex[-1]
            if endIndex >= startIndex:
                notesContainedTmp = np.arange(startIndex, endIndex + 1, 1)

        notesContained.append(notesContainedTmp)

    return notesContained


# TODO static analysis of this function
# Convert magnitudes received from FFT to notes and their magnitudes
# (Returns three arrays: notes, exact notes freqs, magnitude )
def getNoteMagnitudes(ffts, freqs, resolution):
    output = []
    notes, noteFreqs = getNoteFreqs()

    # Edit ffts and freqs to only go to 5kHz. We don't need to
    # iterate over all of them
    freqs = [f for f in freqs if f < 5000]
    ffts = [fft[0: len(freqs)] for fft in ffts]

    notesContained = getNotesContainedInFreqs(noteFreqs, freqs, resolution)

    # For each frame passed to the FFT
    for fft in ffts:
        mags = np.zeros(len(noteFreqs))

        # For each frequency contained in the FFT output
        for i in range(len(freqs)):

            # Add the magnitude to the note frequency (freqs defined by
            # notesContained)
            if len(notesContained[i]) > 0:
                mag = fft[i] / len(notesContained[i])
                # Add to the note magnitude(s)
                for index in notesContained[i]:
                    mags[index] += mag 

        # Append mags for this FFT result (of one frame)
        output.append(mags)

    return notes, noteFreqs, np.array(output)


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
    outputNodes = 144

    model = keras.Sequential()

    model.add(keras.Input(shape=(inputNodes, )))
    model.add(keras.layers.Dense(outputNodes, activation="relu"))

    model.compile(optimizer="adam", loss="mean_squared_error")

    randomInput = np.array([np.random.random(inputNodes)])
    randomPredict = model.predict(randomInput)
    #  print("Input shape:", randomInput.shape, ", output shape:", randomPredict.shape)

    return model

def saveModel(model):
    model.save(modelPath)

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
    #  return major + minor + seventh

def ls(path):
    return subprocess.run(["ls", path], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")[0: -1]



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

    # Create and save a new model if the user wants to
    if sys.argv[1] == "init":
        model = newModel()
        saveModel(model)
        print("New model created and saved. Please run without init now.")
        quit(0)

    if sys.argv[1] == "train":
        train = True
        print("Training started")
        inputFiles = ls(chordsPath)
    else:
        train = False
        inputFiles = [sys.argv[1]]

    # Get chords strings in the same array that the predictions are
    chordsStrings = getChordsStringsArray()

    # Data that will be fed to the NN:
    nnInputs = []
    nnOutputs = []

    for inputFile in inputFiles:
        print("Processing file", inputFiles.index(inputFile) + 1, "of", len(inputFiles))

        # Get the file and convert it to WAV if needed
        if not "wav" in inputFile:
            # If it is a mp3, convert it to a wav
            if "mp3" in inputFile:
                inputFile = inputFile.replace("mp3", "wav")
                AudioSegment.from_mp3(sys.argv[1]).export(inputFile, format = "wav")
            else:
                print("ERROR: The input audio file must be a mp3 or a wav file")
                quit(1)

        # Read the data from the file
        if train:
            path = chordsPath + "/" + inputFile
        else:
            path = audioPath + "/" + inputFile
        sampleRate, samples = wavfile.read(path)
        sampleCount = len(samples)
        audioLength = sampleCount / sampleRate
        resolution = sampleRate / fftWidth

        if not train:
            print("Sample rate: ", sampleRate, ", resolution of the FFT: ", "{:.2f}".format(resolution))
            print("Amount of samples: ", sampleCount, ", audio length [s]: ", "{:.2f}".format(audioLength))

        # If the audio is stereo, convert it to mono
        if len(samples.shape) == 2:
            channels = samples.shape[1]
            samples = np.transpose(samples)
            tmp = np.zeros(len(samples[0]))
            for i in range(channels):
                tmp += samples[i]
            tmp /= channels
            samples = np.transpose(tmp)

        # Plot the input signal
        #  plotSignal(samples, audioLength)

        # Normalize the data
        normSamples = normalizeSamples(samples, sampleCount)

        # Get frames
        frames = getFrames(normSamples, sampleCount)

        # Plot one of the frames
        #  plotSignal(frames[24], audioLength)

        # Discrete fourier transform
        magnitudes, freqs = transformSignal(frames, sampleRate)

        # Normalize the FFT output (energy needs to be preserved)
        # 64 here is just a magic number so the numbers are below 0
        magnitudes /= fftWidth / 64

        # Plot the result
        #  plotDft(freqs, magnitudes[50])

        # Get notes and their magnitudes (2 arrays: note freqs and magnitudes)
        notes, noteFreqs, noteMags = getNoteMagnitudes(magnitudes, freqs, resolution)
        #  print(noteMags[50])

        #  if len(nnInputs) == 0:
            #  nnInputs = noteMags
        #  else:
            #  nnInputs = nnInputs + noteMags
        for i in range(len(noteMags)):
            nnInputs.append(noteMags[i])

        if train:
            chordIndex = chordsStrings.index(inputFile.split("_")[1])
            output = np.zeros(144)
            output[chordIndex] = 1
            for i in range(len(noteMags)):
                nnOutputs.append(output)

    # Load the neural network (tf model)
    model = tf.keras.models.load_model(modelPath)

    # Train
    if train:
        nnInputs = np.array(nnInputs)
        nnOutputs = np.array(nnOutputs)
        print("Training with", len(nnInputs), "inputs and", len(nnOutputs), "outputs")
        # Train the model
        model.fit(nnInputs, nnOutputs, batch_size=100, epochs=10, shuffle=True)
        # Save the model
        saveModel(model)

    # Predict
    else:
        # Pass the noteMags to the neural network to get the chord for each frame
        predictions = model.predict(noteMags)

        # Print the predicted chords
        print("Predictions: ")
        lastIndex = -1
        for i in range(len(predictions)):
            confidence = max(predictions[i])
            chordIndex = np.where(predictions[i] == confidence)[0][0]
            if chordIndex != lastIndex:
                fromTime = sampleToTime(i * fftStep, sampleRate)
                print("From ", "{:.2f}".format(fromTime) + "s:",
                    "chord ", chordsStrings[chordIndex].ljust(7), 
                    " with confidence of ", str(int(confidence * 100)) + "%")
                lastIndex = chordIndex






###############
## MAIN CALL ##
###############

if __name__ == "__main__":
    main();
