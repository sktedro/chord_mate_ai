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
	python3 ${SCRIPT} test

clean:
	rm -rf __pycache__ data/*_backup.npz
