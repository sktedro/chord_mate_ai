# All settings in one place

###########
## PATHS ##
###########

# Training .wav files are expected in chordsPath directory and with names like
# chord_A5_0.wav, chord_E_1.wav, ...
chordsPath = "./audio/chords"

notesPath = "./audio/notes"

audioPath = "./audio"

dataDir = "./data"
trainingDataFileName = "training_data.npz"
testingDataFileName = "testing_data.npz"
# If the data file already exists, it will be loaded and new data will be 
# appended. Otherwise, it will be created

modelPath = "./nn/"

######################
## AUDIO PROCESSING ##
######################

fftWidth = 8192
#  fftStep = 2048
fftStep = 4096
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

# Network layers and amounts of nodes they consist of
# By the end of one training: 0.0043, converging just a tiny bit
#  nnNodes = [96, 576, 144]
# By the end of one training: 0.0042, not really converging any more
#  nnNodes = [96, 2048, 144]
# By the end of one training: 0.0045, not really converging any more
#  nnNodes = [96, 288, 144]
# By the end of one training: 0.0030, 
# Performing okay after a lot of training (loss around 8e-04)
#  nnNodes = [96, 512, 256, 144]
# By the end of one training: 0.0023
#  nnNodes = [96, 192, 576, 288, 144]
# By the end of one training: 0.0024, converging just a tiny bit
#  nnNodes = [96, 192, 256, 288, 144]
# By the end of one training: 0.0015, stil converging slowly
#  nnNodes = [96, 192, 256, 256, 288, 144]
# By the end of one training: 0.0010, still converging slowly
# Performing very well after just an hour of training (loss around 2-3e-04)
nnNodes = [96, 192, 384, 1024, 576, 288, 144]
# By the end of one training: 0.0021, converging
#  nnNodes = [96, 384, 512, 1024, 1024, 576, 288, 144]
# By the end of one training: 0.0064, converging
# Loss around 5e-04 after an hour of training
#  nnNodes = [96, 384, 1024, 2048, 1024, 576, 288, 144]

# Epochs to train for in one training cycle
trainingEpochs = 5

# This divides the training data to chunks of size inputsPerTraining
# Lower this number if the RAM limit is getting exceeded
inputsPerTraining = 100000

# Number of training inputs after which the model should be recalculated
batchSize = 128
