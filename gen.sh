#!/bin/bash

mkdir continue_generating

while [ -d continue_generating ]; do
  echo "Generating"
  for i in {1..10}; do 
    python3 generate_notes.py 25 $(($RANDOM % 8 + 3)) >/dev/null & 
  done

  for job in `jobs -p`; do
    echo "Waiting for $job"
    wait $job
  done
  echo "Done generating"

  python3 prepare_data.py train notes
  # python3 prepare_data.py test notes

  rm -rf audio/notes_generated/*
  echo "Done preparing the data"
done
