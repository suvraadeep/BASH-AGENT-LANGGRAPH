#!/bin/bash

directory=/workspace
pattern=*.c
min_size=1024
find "$directory" -type f -name "$pattern" -size +$min_size