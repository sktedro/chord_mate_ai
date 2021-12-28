#!/bin/bash

AMOUNT=100

while true; do 
  INSTRUMENTS=$(($RANDOM % 7 + 1))
  echo "Generating $AMOUNT chords with $INSTRUMENTS instruments per chord"
  python3 gen_chords.py "$AMOUNT" "$INSTRUMENTS"
done
