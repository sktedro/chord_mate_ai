import os
from pydub import AudioSegment 
import sys
import numpy as np
#  import matplotlib.pyplot as plt
import wave
import subprocess

import misc
import settings
import fourier_transform

###############
## FUNCTIONS ##
###############

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

    # Perform the fourier transform
    freqs, magnitudes, sampleRate = fourier_transform.fourier_transform(path, verbose)

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
