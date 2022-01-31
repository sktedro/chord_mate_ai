#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Please provide a target for the data: train/test and how many processes to run"
  exit 0
fi

echo "Starting. To stop, please remove the 'continue generating' file and wait"

touch continue_generating

while [ -f continue_generating ]; do
  echo "Generating"
  for i in $(seq 1 $2); do 
    python3 generate_chords.py 10 $(($RANDOM % 10 + 1)) >/dev/null & 
    echo "Starting process $!"
  done

  for job in `jobs -p`; do
    echo "Waiting for process $job"
    wait $job
  done
  echo "Done generating"

  python3 prepare_data.py "$1" chords

  rm audio/chords/*
  echo "Done preparing the data"
done
