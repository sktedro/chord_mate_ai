SCRIPT=chord_mate_ai.py

all:
	@echo This target does nothing, for now

sample: 
	python3 ${SCRIPT} sample.wav

init: 
	python3 ${SCRIPT} init
