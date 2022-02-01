# All settings in one place

###########
## PATHS ##
###########

# Training .wav files are expected in chordsPath directory and with names like
# chord_A5_0.wav, chord_E_1.wav, ...
# Do not change this if you don't know what you're doing!
chordsPath = "./audio/chords"

# Do not change this if you don't know what you're doing!
notesPath = "./audio/notes"

# Do not change this if you don't know what you're doing!
audioPath = "./audio"

# Do not change these if you don't know what you're doing!
dataDir = "./data"
trainingDataFileName = "training_data"
testingDataFileName = "testing_data"
# If the data file already exists, it will be loaded and new data will be 
# appended. Otherwise, it will be created
# To test with the training data, uncomment this:
#  testingDataFileName = "training_data"

modelPath = "./nn"

######################
## AUDIO PROCESSING ##
######################

fftWidth = 4096
#  fftStep = 2048
fftStep = 2048
# With sample rate 44100, this makes the FFT accuracy around 5Hz
# TODO Make it adaptive to the sampling frequency? Or don't accept files with
# Fs other than 44100?

######################
## CHORD GENERATION ##
######################

# Orders of notes to use when constructing chords
minOrder = 2
maxOrder = 5

# If set to true, instruments playing chords may differ (if chordsAmount > 1)
# (the generated audio could consist of chords played by different instruments)
mixInstruments = True

# Probability that a note in a chord will be there twice (second time with
# higher order)
noteDuplicateProbability = 0.5

##############
## TRAINING ##
##############

# Settings 1: 
# 67.25%, 70.89%
#  hiddenLayersActivationFn = "tanh"
#  outputLayerActivationFn = "tanh"
#  optimizer = "adam"
#  lossFunction = "mean_squared_error"

# Settings 2: 
# 85.84%, 87.48%, 87.30%, 88.38%, 87.62%
hiddenLayersActivationFn = "sigmoid"
outputLayerActivationFn = "sigmoid"
optimizer = "adam"
lossFunction = "binary_crossentropy"

# Settings 3: 
# 84.68%, 85.54%, 85.55%, 85.81%, 85.56%, 86.61%
#  hiddenLayersActivationFn = "tanh"
#  outputLayerActivationFn = "softmax"
#  optimizer = "adam"
#  lossFunction = "categorical_crossentropy"

# Layers:
# Input layer:
    # Inputs: Magnitudes of tones (A0, A#0, ..., G#0, A1, ..., G#7)
    # Inputs amount: 12 * 8 = 96
# Output layer:
    # Outputs: Chords (A, A#, ..., G)(major, minor, 7, 5, ...)
    # Outputs amount: 12 * 12 = 144

# 92.3%
nnNodes = [96, 256, 256, 144]
# 91.4%
#  nnNodes = [96, 256, 512, 512, 256, 144]

# Epochs to train for in one training cycle
trainingEpochs = 1

# This divides the training data to chunks of size inputsPerTraining
# Lower this number if the RAM limit is getting exceeded
inputsPerTraining = 100000

# Number of training inputs after which the model should be recalculated
batchSize = 32
