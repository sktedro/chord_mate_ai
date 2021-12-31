# All settings in one place

# Sampling frequency
fs = 44100

###########
## PATHS ##
###########

# Training .wav files are expected in chordsPath directory and with names like
# chord_A5_0.wav, chord_E_1.wav, ...
# Do not change this if you don't know what you're doing!
chordsPath = "./audio/chords"
generatedNotesPath = "./audio/notes_generated"

# Do not change this if you don't know what you're doing!
notesPath = "./audio/notes"

# Do not change this if you don't know what you're doing!
audioPath = "./audio"

# Do not change these if you don't know what you're doing!
dataDir = "./data"
noteDetDataDir = "./data_note_detection"
trainingDataFileName = "training_data"
testingDataFileName = "testing_data"

#  trainingDataFileName = "testing_data"
#  testingDataFileName = "training_data"

# If the data file already exists, it will be loaded and new data will be 
# appended. Otherwise, it will be created
# To test with the training data, uncomment this:
#  testingDataFileName = "training_data"

modelPath = "./nn"
noteDetModelPath = "./nn_note_detection"

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

#####################
## NOTE GENERATION ##
#####################

minNoteOrder = 0
maxNoteOrder = 7

noteGenMixInstruments = True

##############
## TRAINING ##
##############

# General settings

threadLimit = 7


# Note detection settings

noteDetHiddenLayersActivationFn = "sigmoid"
noteDetOutputLayerActivationFn = "sigmoid"

noteDetOptimizer = "adam"

#  noteDetLossFunction = "mean_squared_error"
noteDetLossFunction = "binary_crossentropy"

#  noteDetNnNodes = [816, 256, 12]
noteDetNnNodes = [816, 1024, 128, 12]

# Epochs to train for in one training cycle
noteDetTrainingEpochs = 5

# This divides the training data to chunks of size inputsPerTraining
# Lower this number if the RAM limit is getting exceeded
noteDetInputsPerTraining = 500000

# Number of training inputs after which the model should be recalculated
noteDetBatchSize = 32


# Chord detection settings

hiddenLayersActivationFn = "tanh"
outputLayerActivationFn = "tanh"
optimizer = "adam"
lossFunction = "mean_squared_error"

# Better settings for categorization
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

# Best: 90.3%
#  nnNodes = [96, 256, 512, 256, 144]

nnNodes = [96, 512, 1024, 512, 144]

# Achieved 90% accuracy
#  nnNodes = [96, 192, 384, 1024, 576, 288, 144]

# Epochs to train for in one training cycle
trainingEpochs = 3

# This divides the training data to chunks of size inputsPerTraining
# Lower this number if the RAM limit is getting exceeded
inputsPerTraining = 500000

# Number of training inputs after which the model should be recalculated
#  [512 1024 2048]. Others seem to converge
batchSize = 32
