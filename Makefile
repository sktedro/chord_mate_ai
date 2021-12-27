SCRIPT=chord_mate_ai.py

all:
	@echo This target does nothing, for now

sample: 
	python3 ${SCRIPT} ./audio/sample.wav

sample2: 
	python3 ${SCRIPT} ./audio/sample2.wav

train:
	python3 ${SCRIPT} train

init: 
	python3 ${SCRIPT} init

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
