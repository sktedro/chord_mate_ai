#!/bin/bash

# Parses data files in midi-js-soundfonts dataset so they can be used by
# gen_chords.py to generate chords
# Careful, from 9.2GB it makes the dataset TODO (by converting to .wav)

DS1NAME="FatBoy"
DS2NAME="FluidR3_GM"
DS3NAME="MusyngKite"

# Process the first argument (path to the midi-js-soundfonts contents)
if [ -z "$1" ]; then
  echo "Please pass the path to the dataset folder as an argument"
  exit 0
fi
if [ ! -d "$1" ]; then
  echo "Dataset directory not found"
  exit 0
fi
DSDIR="$1"
if [ "${DSDIR: -1}" = "/" ]; then
  DSDIR="${DSDIR%?}"
fi

# Get the destination directory
echo "Please enter a path to use for the destination directory (eg. './audio/notes/')"
read NEWDIR
if [ "${NEWDIR: -1}" = "/" ]; then
  NEWDIR="${NEWDIR%?}"
fi
echo "Chosen $NEWDIR"

process_folder(){
  SRCDIR="$1"
  DSNAME="$2"
  NEWDIR="$3"
  NUM="$4"

  echo "Processing sub-dataset: $DSNAME"

  # Sub-datasets will be called datasets from now on
  DS="$DSDIR/$DSNAME"

  # Check if the directory exist
  if [ ! -d "$DS" ]; then
    echo "Folder $DS not found in $1. Is the dataset missing?"
    exit 0
  fi

  # Copy the dataset to a temporary directory
  echo "Copying data"
  cp -r "$DS" "$NEWDIR"

  DS="$NEWDIR/$DSNAME"

  # Remove the .js and .js.gz files
  rm -f "$DS/"*".js" 
  rm -f "$DS/"*".js.gz"
  rm -f "$DS/"*".json"

  # For all target directories
  for dir in "$DS/"*; do
    echo "Processing dir: $dir"

    # Add a number to all the file names
    for file in "$dir/"*; do
      # echo "Processing file: $file"
      # Remove the extension
      f="${file/.mp3/}"
      # Add the number and the extension back
      f="$f""_""$NUM"".mp3"
      mv "$file" "$f"
    done

    # Convert all files to .wav and remove the .mp3 (old) ones
    for file in "$dir/"*; do
      ffmpeg -loglevel error -i "$file" "${file/.mp3/.wav}"
      rm "$file"
    done

    # Rename the directory to discard the "-mp3" part
    mv "$dir" "${dir/-mp3/}"

  done

  # Move the contents of the sub-dataset to the NEWDIR
  cp -r "$DS/"* "$NEWDIR"
  rm -rf "$DS"
}

mkdir -p "$NEWDIR"

process_folder "$DSDIR" "$DS1NAME" "$NEWDIR" "0"
process_folder "$DSDIR" "$DS2NAME" "$NEWDIR" "1"
process_folder "$DSDIR" "$DS3NAME" "$NEWDIR" "2"

# Move short ones into their own folder (0_short)
mkdir "$NEWDIR/0_short"
mv "$NEWDIR/acoustic_bass"         "$NEWDIR/0_short"
mv "$NEWDIR/agogo"                 "$NEWDIR/0_short"
mv "$NEWDIR/banjo"                 "$NEWDIR/0_short"
mv "$NEWDIR/dulcimer"              "$NEWDIR/0_short"
mv "$NEWDIR/electric_guitar_muted" "$NEWDIR/0_short"
mv "$NEWDIR/glockenspiel"          "$NEWDIR/0_short"
mv "$NEWDIR/guitar_fret_noise"     "$NEWDIR/0_short"
mv "$NEWDIR/kalimba"               "$NEWDIR/0_short"
mv "$NEWDIR/koto"                  "$NEWDIR/0_short"
mv "$NEWDIR/marimba"               "$NEWDIR/0_short"
mv "$NEWDIR/melodic_tom"           "$NEWDIR/0_short"
mv "$NEWDIR/music_box"             "$NEWDIR/0_short"
mv "$NEWDIR/orchestra_hit"         "$NEWDIR/0_short"
mv "$NEWDIR/orchestral_harp"       "$NEWDIR/0_short"
mv "$NEWDIR/pad_6_metallic"        "$NEWDIR/0_short"
mv "$NEWDIR/pizzicato_strings"     "$NEWDIR/0_short"
mv "$NEWDIR/reverse_cymbal"        "$NEWDIR/0_short"
mv "$NEWDIR/shamisen"              "$NEWDIR/0_short"
mv "$NEWDIR/sitar"                 "$NEWDIR/0_short"
mv "$NEWDIR/steel_drums"           "$NEWDIR/0_short"
mv "$NEWDIR/synth_bass_1"          "$NEWDIR/0_short"
mv "$NEWDIR/synth_bass_2"          "$NEWDIR/0_short"
mv "$NEWDIR/synth_drum"            "$NEWDIR/0_short"
mv "$NEWDIR/taiko_drum"            "$NEWDIR/0_short"
mv "$NEWDIR/timpani"               "$NEWDIR/0_short"
mv "$NEWDIR/tinkle_bell"           "$NEWDIR/0_short"
mv "$NEWDIR/tubular_bells"         "$NEWDIR/0_short"
mv "$NEWDIR/vibraphone"            "$NEWDIR/0_short"
mv "$NEWDIR/woodblock"             "$NEWDIR/0_short"
mv "$NEWDIR/xylophone"             "$NEWDIR/0_short"
