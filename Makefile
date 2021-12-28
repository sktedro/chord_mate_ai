SCRIPT=chord_mate_ai.py
GENSCRIPT=gen_random.sh
PREPSCRIPT=prepare_data.py

# TODO If there is a file in the first argument, run prediction. Otherwise,
# print usage of this makefile
all:
	@echo This target does nothing, for now

# Initialize a neural network
init: 
	python3 ${SCRIPT} init

# Generate chords from notes (random number of instruments, in an infinite loop)
gen:
	./${GENSCRIPT}

# Prepare the training data from all chords generated 
prep_train: 
	python3 ${PREPSCRIPT} train

# Prepare the testing data from all chords generated 
prep_test: 
	python3 ${PREPSCRIPT} test

# Train the neural network with prepared training data until stopped by user
train:
	while true; do python3 ${SCRIPT} train; done

# TODO
test:
	python3 ${SCRIPT} "./audio/chords_test/chord_A_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_A#_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_B_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_C_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_C#_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_D_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_D#_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_E_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_F_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_F#_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_G_0.wav"
	python3 ${SCRIPT} "./audio/chords_test/chord_G#_0.wav"

clean:
	rm __pycache__
