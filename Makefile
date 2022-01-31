SCRIPT=chord_mate_ai.py
GENSCRIPT=generate_data.sh
PREPSCRIPT=prepare_data.py

# TODO If there is a file in the first argument, run prediction. Otherwise,
# print usage of this makefile
all:
	@echo This target does nothing, for now

# Initialize a neural network
init: 
	python3 ${SCRIPT} init

# Generate training data
gen_train:
	./${GENSCRIPT} train 10

# Generate testing data
gen_test:
	./${GENSCRIPT} test 10

# Train the neural network with prepared training data until stopped by user
train:
	while true; do python3 ${SCRIPT} train; done

# TODO
test:
	python3 ${SCRIPT} test

clean:
	rm -rf __pycache__ data/*_backup.npz
