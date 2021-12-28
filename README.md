# Brief

The idea is a script to which the user inputs an audio file and receives chords
extracted from it. 

As I tried to do this using javascript, I learned that JS is really not made
for this, so this repository is my attempt to use python instead. However,
note that this project may never be finished.

JS attempt: https://github.com/sktedro/chord_recognizer


# Configuration and settings

To change settings, edit the `settings.py` file


# Running

## Neural network initialization

If you want to create your own neural network, first you need to initialize an
empty one. This will overwrite the neural network saved in `nn/`

`make init`

## Generating chords

To generate the training data, you first need chords to generate it from.
Chords are `.wav` or `.mp3` files containing a sound of a chord, while the files
name represents the chord. Amount of instruments, duplicate notes in a chord,
... can vary

I strongly recommend the midi-js-soundfonts note dataset. For my purposes, I
combined `FatBoy`, `FluidR3_GM` and `MusyngKite` into one folder and moved all
instruments of which sound is shorter than the file (eg. drums make sound for a
split second while the file is 3 seconds long) to `0_short` folder, which is 
then discarded by the chord generator.

Make sure you have:
 - `./audio/notes/` folder
 - folders for different instruments in that folder
 - in each instrument folder, `.wav` or `.mp3` files with name `A0_x` while `A` 
   stands for a tone, `0` for the order of the tone and `x` just to distinguish 
   different files playing the same note (useful if you have notes captured by 
   mic and pickup (for example))

`python3 gen_chords.py amount instrumentsAmount`

instrumentsAmount stands for the amount of instruments to use when generating 
the chord (each instrument plays the chord separately, but audio is merged)


## Generating training data

To train the neural network, we need the training data. This is generated by
processing all chord files in `./audio/chords/` directory and is then saved to
`./training_data/training_data.npz` (the file is compressed and represents 
inputs and ouputs in a numpy array)

Make sure you have:
 - `./audio/chords/` folder
 - chords in that folder (`.wav` or `.mp3`) with names `chord_C_x` while `C` 
   stands for a chord name (C, C#, A7, ...) and `x` is there just to distinguish 
   different files playing the same chord (amount of instruments, notes, ... 
   can vary)

`make prep`

## Training

Make sure you have:
 - training data generated in `training_data/` folder
 - a neural network initialized

`make train`

## Predicting (find chords in a .wav or .mp3 file)

To recognize chords in a `.wav` or `.mp3` file:

Make sure you have:
 - a trained neural network

`python3 chord_mate_ai.py path/to/your/file.wav`


# Folder structure

 - `chord_mate_ai.py` - main program to run when training or predicting
 - `settings.py` - global variables as settings. All configuration should be 
   done here
 - `gen_chords.py` - generate chords from notes
 - `prepare_data.py` - call process_audio.py for every chord in 
   `./audio/chords/` and by that, create inputs and outputs for the neural 
   network. Then write that data to `./training_data/`
 - `process_audio.py` - create and return inputs and outputs for the neural 
   network. If not training, outputs will be an empty array.
 - `./audio/` - all audio files
 - `./audio/notes/` - notes to generate chords from. Format is explained in
   section Running, Generating chords
 - `./audio/chords/` - chords that should be processed to create the training
   data. Format is exampled in section Running, Generating training data
 - `./nn/` - the neural network is saved in here
 - `./training_data/` - training data in format `.npz`. More about that in 
   section Running, Generating training data


# Datasets used (maybe not all of them, but I'd like to mention them all):
 - https://magenta.tensorflow.org/datasets/nsynth
 - https://github.com/gleitz/midi-js-soundfonts
