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

modelPath = "./nn/"

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
minOrder = 0
maxOrder = 7

# If set to true, instruments playing chords may differ (if chordsAmount > 1)
# (the generated audio could consist of chords played by different instruments)
mixInstruments = True

# Probability that a note in a chord will be there twice (second time with
# higher order)
noteDuplicateProbability = 0.5

##############
## TRAINING ##
##############

hiddenLayersActivationFn = "tanh"
#  outputLayerActivationFn = "tanh"
optimizer = "adam"
#  lossFunction = "mean_squared_error"

# Better settings for categorization
#  hiddenLayersActivationFn = "tanh"
outputLayerActivationFn = "softmax"
#  optimizer = "adam"
lossFunction = "categorical_crossentropy"


# Best: 66%
#  nnNodes = [96, 144]

#  nnNodes = [96, 512, 144]

# Achieved 74% accuracy
nnNodes = [96, 192, 384, 1024, 576, 288, 144]

# Epochs to train for in one training cycle
trainingEpochs = 3

# This divides the training data to chunks of size inputsPerTraining
# Lower this number if the RAM limit is getting exceeded
inputsPerTraining = 100000

# Number of training inputs after which the model should be recalculated
#  [512 1024 2048]. Others seem to converge
batchSize = 32
