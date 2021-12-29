import os
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal # TODO
from math import e # TODO
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib # TODO
import wave
import subprocess
import settings

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
    for i in range(sampleCount // settings.fftStep - 1):
        frames.append(samples[i * settings.fftStep : i * settings.fftStep + settings.fftWidth])
    #  print("We have", len(frames), "frames, all with length", len(frames[0]))
    return frames

def transformSignal(frames, sampleRate):
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: settings.fftWidth // 2] for k in range(len(frames))]))
    #  magnitudes = abs(np.array([
        #  np.fft.fft(a = frames[k], n = sampleRate) for k in range(len(frames))]))
    freqs = [k * sampleRate // settings.fftWidth for k in range(settings.fftWidth // 2)]
    return magnitudes, freqs

#  def transformSignal(frames, sampleRate):
    #  #  magnitudes = abs(np.array([np.fft.fft(frames[k])[0: settings.fftWidth // 2] for k in range(len(frames))]))

    #  freq = 1000
    #  samples = sampleRate / 4
    #  f = np.arange(0, samples) / sampleRate * np.pi * 2
    #  cos = np.cos(f * freq)

    #  #  plt.plot(cos)

    #  fft1 = abs(np.fft.fft(cos))
    #  plt.plot(np.arange(0, sampleRate, 4), fft1)

    #  fft2 = abs(np.fft.fft(cos * np.hanning(len(cos)), n = sampleRate))
    #  plt.plot(np.arange(0, sampleRate, 1), fft2)

    #  fft3 = abs(np.fft.fft(np.cos(freq * np.arange(0, sampleRate) / sampleRate * np.pi * 2)))
    #  plt.plot(fft3)

    #  plt.show()


    #  #  frame = frames[5]
    #  #  a = abs(np.fft.fft(a = frame, n = sampleRate))
    #  #  #  a = abs(np.fft.fft(a = np.pad(frame, (0, sampleRate - len(frame)))))
    #  #  tmp = abs(np.fft.fft(a = frame))

    #  #  b = []
    #  #  m = len(a) / len(tmp)
    #  #  print(m)
    #  #  for i in range(len(a)):
        #  #  if i % int(m) == 0 and i // m < len(tmp):
            #  #  b.append(tmp[int(np.round(i / m))])
        #  #  else:
            #  #  b.append(0)

    #  #  plt.plot(b)
    #  #  plt.plot(a)
    #  #  plt.show()

    #  quit(0)
    #  freqs = [k * sampleRate // settings.fftWidth for k in range(settings.fftWidth // 2)]
    #  return magnitudes, freqs


def plotDft(freqs, magnitudes):
    plt.plot(freqs, magnitudes)
    plt.title("Discrete fourier transform")
    plt.ylabel("Magnitude []")
    plt.xlabel("Frequency [Hz]")
    plt.show()

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

def processAudio(path, training, verbose):
    fileName = path.split("/")[-1]

    # Get chords strings in the same array that the predictions are
    chordsStrings = getChordsStringsArray()

    # Arrays where the acquired data will be written
    nnInputs = []
    nnOutputs = []

    # Get the file and convert it to WAV if needed
    if not "wav" in fileName:
        # If it is a mp3, convert it to a wav
        if "mp3" in fileName:
            fileName = fileName.replace("mp3", "wav")
            AudioSegment.from_mp3(sys.argv[1]).export(fileName, format = "wav")
        else:
            print("ERROR: The input audio file must be a mp3 or a wav file")
            quit(1)

    sampleRate, samples = wavfile.read(path)
    sampleCount = len(samples)
    audioLength = sampleCount / sampleRate
    resolution = sampleRate / settings.fftWidth + 1 # + 1 for some tolerance

    if verbose:
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

    # Normalize the data
    normSamples = normalizeSamples(samples, sampleCount)

    # Get frames
    frames = getFrames(normSamples, sampleCount)

    # Discrete fourier transform
    magnitudes, freqs = transformSignal(frames, sampleRate)

    # Normalize the FFT output (energy needs to be preserved)
    # 64 here is just a magic number so the numbers are below 0
    magnitudes /= settings.fftWidth / 64

    # Plot the result
    #  plotDft(freqs, magnitudes[5])

    # Get notes, exact notes freqs and magnitudes
    notes, noteFreqs, noteMags = getNoteMagnitudes(magnitudes, freqs, resolution)

    # Append the magnitudes to nnInputs
    for i in range(len(noteMags)):
        nnInputs.append(noteMags[i])

    if training:
        chordIndex = chordsStrings.index(fileName.split("_")[1])
        output = np.zeros(144)
        output[chordIndex] = 1.0
        for i in range(len(noteMags)):
            nnOutputs.append(output)

    nnInputs = np.array(nnInputs)
    nnOutputs = np.array(nnOutputs)

    # This causes the NN to be less accurate, for some reason
    # (it is more confident with wrong outputs)
    # Normalize the NN inputs
    #  for i in range(len(nnInputs)):
        #  multiplier = 1 / max(nnInputs[i])
        #  nnInputs[i] *= multiplier
        #  TODO also append the multiplier to the nn input (maybe inverted or
        #  something)

    return nnInputs, nnOutputs, sampleRate
