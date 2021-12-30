"""
Input: Path to a file, verbosity boolean
Output: Frequencies and their magnitudes, sample rate
"""

from scipy.io import wavfile
import numpy as np
import settings

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
    #  print("We have", len(frames), "frames, all with length", settings.fftWidth)
    return frames

def transformSignal(frames, sampleRate):

    magnitudes = []
    trim = sampleRate // 2
    han = np.hanning(settings.fftWidth)

    for k in range(len(frames)):
        a = frames[k] * han
        fft = np.fft.fft(a = a , n = sampleRate)[0: trim]
        magnitudes.append(abs(np.array(fft)))
    magnitudes = np.array(magnitudes)

    freqs = np.arange(0, trim)

    return magnitudes, freqs

def plotDft(freqs, magnitudes):
    plt.plot(freqs, magnitudes)
    plt.title("Discrete fourier transform")
    plt.ylabel("Magnitude []")
    plt.xlabel("Frequency [Hz]")
    plt.show()

def fourier_transform(samples, sampleRate, verbose):
    sampleCount = len(samples)
    audioLength = sampleCount / sampleRate
    resolution = sampleRate / settings.fftWidth + 1 # + 1 for some tolerance

    if verbose:
        print("Sample rate: ", sampleRate, ", resolution of the FFT around: ", "{:.2f}".format(resolution))
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

    return freqs, magnitudes
