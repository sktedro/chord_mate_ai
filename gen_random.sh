#!/bin/bash

while true; do 
  i=$(($RANDOM % 7 + 1))
  echo "$i instruments per chord"
  python3 gen_chords.py 100 "$i"
done
