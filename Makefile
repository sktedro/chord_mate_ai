SCRIPT=chord_mate_ai.py

all:
	@echo This target does nothing, for now

sample: 
	python3 ${SCRIPT} sample.wav

sample2: 
	python3 ${SCRIPT} sample2.wav

train:
	python3 ${SCRIPT} train


init: 
	python3 ${SCRIPT} init
