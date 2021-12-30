import os
from pydub import AudioSegment 
import sys
import numpy as np
import matplotlib.pyplot as plt
import wave
import subprocess
from scipy import signal
from pydub.playback import play

import pyloudnorm as pyln

import misc
import settings
import fourier_transform

###############
## FUNCTIONS ##
###############

def plotSpectrogram(frequencies, spectrogram, audioLength):
    time = np.arange(0, audioLength, audioLength / len(spectrogram[0]))
    print(len(spectrogram[0]))
    plt.pcolormesh(time, frequencies, spectrogram, shading='gouraud')
    plt.show()

def getSpectrogram(magnitudes):
    return np.transpose(10 * np.log10(abs(magnitudes) ** 2))


def cropByLoudness(inputSignal, verbose):
    sampleSize = 1024

    threshold = -65

    meter = pyln.Meter(sampleSize)
    samples = []
    zeroPaddedSignal = np.concatenate((np.array(inputSignal), np.zeros(sampleSize)))
    for i in np.arange(0, len(inputSignal), sampleSize):
        if i < len(zeroPaddedSignal) - sampleSize:
            samples.append(zeroPaddedSignal[i: i + sampleSize])

    zeroPaddedSignal = np.array(zeroPaddedSignal / max(abs(zeroPaddedSignal))).astype(np.float32)

    loudness = []
    for sample in samples:
        loudness.append(meter.integrated_loudness(sample))

    quietSampleIndex = np.where(np.array(loudness) < threshold)[0]
    if len(quietSampleIndex) > 0:
        quietSampleIndex = quietSampleIndex[0]
        outputSignal = inputSignal[: sampleSize * quietSampleIndex]
        if verbose:
            print("Cropping", quietSampleIndex + 1, "samples from", len(loudness), "total")
    else:
        outputSignal = inputSignal

    return outputSignal

def getNoteMagnitudes(ffts, freqs):
    output = []
    notes, noteFreqs = misc.getNoteFreqs()
    noteFreqs = np.rint(noteFreqs).astype(int)

    # Edit ffts and freqs to only go to 5kHz. We don't need to
    # iterate over all of them
    freqs = [f for f in freqs if f < 5000]
    ffts = [fft[0: len(freqs)] for fft in ffts]

    # For each frame passed to the FFT
    for fft in ffts:
        mags = []
        for freq in noteFreqs:
            mags.append(fft[int(freq)])

        # Append mags for this FFT result (of one frame)
        output.append(mags)

    return notes, noteFreqs, np.array(output)


def getFileName(path):
    fileName = path.split("/")[-1]
    # Get the file and convert it to WAV if needed
    if not "wav" in fileName:
        # If it is a mp3, convert it to a wav
        if "mp3" in fileName:
            fileName = fileName.replace("mp3", "wav")
            AudioSegment.from_mp3(sys.argv[1]).export(fileName, format = "wav")
        else:
            print("ERROR: The input audio file must be a mp3 or a wav file")
            quit(1)
    return fileName

def processAudio(path, training, verbose):
    # Get the file name and convert to wav if needed
    fileName = getFileName(path)

    # Get chords strings in an array shaped the same as predictions
    chordsStrings = misc.getChordsStringsArray()

    # Read the wav file
    sampleRate, samples = wavfile.read(path)

    # Perform the fourier transform
    freqs, magnitudes = fourier_transform.fourier_transform(samples, sampleRate, verbose)

    # Get notes, exact notes freqs and magnitudes
    notes, noteFreqs, noteMags = getNoteMagnitudes(magnitudes, freqs)

    # Arrays where the acquired data will be written
    nnInputs = noteMags

    nnOutputs = []
    if training:
        chordIndex = chordsStrings.index(fileName.split("_")[1])
        output = np.zeros(144)
        output[chordIndex] = 1.0
        for i in range(len(noteMags)):
            nnOutputs.append(output)

    nnInputs = np.array(nnInputs)
    nnOutputs = np.array(nnOutputs)

    return nnInputs, nnOutputs, sampleRate
