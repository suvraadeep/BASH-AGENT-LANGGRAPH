#!/bin/bash

# Calculate the md5 sum of the contents of the sorted list of files "$FILES"

# Check if FILES is set
if [ -z "$FILES" ]; then
  echo "Error: FILES variable is not set"
  exit 1
fi

# Split FILES into an array
IFS="," read -r -a file_array <<< "$FILES"

# Check if files exist
for file in "${file_array[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Error: File $file does not exist"
    exit 1
  fi
-done

# Calculate md5 sum of file contents
sorted_files=$(printf '%s
' "${file_array[@]}" | sort)
md5_sum=$(md5sum ${sorted_files} | awk '{print $1}')

# Output the md5 sum
echo "$md5_sum"