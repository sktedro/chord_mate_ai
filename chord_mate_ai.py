from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from math import e
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wave

fftWidth = 4096
fftStep = 2048
# With sample rate 44100, this makes the FFT accuracy around 10Hz

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

##########
## MAIN ##
##########

def main():
    print("==================================================")

    ## Read the input audio file

    fileName = sys.argv[1]
    if not "wav" in fileName:
        # If it is a mp3, convert it to a wav
        if "mp3" in fileName:
            fileName = fileName.replace("mp3", "wav")
            AudioSegment.from_mp3(sys.argv[1]).export(fileName, format = "wav")
        else:
            print("ERROR: The input audio file must be a mp3 or a wav file")
            quit(1)

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
