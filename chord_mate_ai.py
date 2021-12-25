import os
# Disable the GPU because of tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
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

fftWidth = 8192
fftStep = 2048
# With sample rate 44100, this makes the FFT accuracy around 5Hz


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
    print("We have", len(frames), "frames, all with length", len(frames[0]))
    return frames

def transformSignal(frames, sampleRate):
    print("Calculating the DFT")
    # FFT
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: fftWidth // 2] for k in range(len(frames))]))
    frequencies = [k * sampleRate // fftWidth for k in range(fftWidth // 2)]
    print("DFT calculated")
    return magnitudes, frequencies

def plotDft(frequencies, magnitudes):
    plt.plot(frequencies, magnitudes)
    plt.title("Discrete fourier transform")
    plt.ylabel("Magnitude []")
    plt.xlabel("Frequency [Hz]")
    plt.show()

def sampleToTime(sampleNum, sampleRate):
    return sampleNum / sampleRate

def plotSignal(samples, duration):
    plt.plot(np.arange(0, duration, duration / len(samples)), samples)
    #  plt.ylabel("Amplitude []")
    #  plt.xlabel("Time [s]")
    #  plt.title("One frame")
    plt.show()


# TODO Convert magnitudes received from FFT to tones and their magnitudes
# (Returns a 2D array: [ exact tone frequency, magnitude ])


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
    model.save("./nn/")


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

    # Get the file and convert it to WAV if needed
    fileName = sys.argv[1]
    if not "wav" in fileName:
        # If it is a mp3, convert it to a wav
        if "mp3" in fileName:
            fileName = fileName.replace("mp3", "wav")
            AudioSegment.from_mp3(sys.argv[1]).export(fileName, format = "wav")
        else:
            print("ERROR: The input audio file must be a mp3 or a wav file")
            quit(1)

    # Read the data from the file
    sampleRate, samples = wavfile.read("./audio/" + fileName)
    sampleCount = len(samples)
    audioLength = sampleCount / sampleRate

    print("Sample rate: ", sampleRate)
    print("Amount of samples: ", sampleCount)
    print("Audio length [s]: ", audioLength)

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
    magnitudes, frequencies = transformSignal(frames, sampleRate)

    # Plot the result
    #  plotDft(frequencies, magnitudes[50])



###############
## MAIN CALL ##
###############

if __name__ == "__main__":
    main();
