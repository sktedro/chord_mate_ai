
fftWidth = 8192
#  fftStep = 2048
fftStep = 4096
# With sample rate 44100, this makes the FFT accuracy around 5Hz
# TODO Make it adaptive to the sampling frequency? Or don't accept files with
# Fs other than 44100?

# Training .wav files are expected in chordsPath directory and with names like
# chord_A5_0.wav, chord_E_1.wav, ...
chordsPath = "./audio/chords"

notesPath = "./audio/notes"

audioPath = "./audio"

trainingDataDir = "./training_data"
trainingDataFileName = "training_data.npz"
# If the file already exists, it will be loaded and new data will be appended
# Otherwise, it will be created

# Orders of notes to use when constructing chords
minOrder = 0
maxOrder = 7

# If set to true, instruments playing chords may differ (if chordsAmount > 1)
# (the generated audio could consist of chords played by different instruments)
mixInstruments = True

# Amount of chords that the output should consist of
chordsAmount = 5

# Probability that a note in a chord will be there twice (second time with
# higher order)
noteDuplicateProbability = 0.5

