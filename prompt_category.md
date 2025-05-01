```bash
#!/bin/bash

# Set source and destination folders
SRC_DIR=/path/to/folderA
DEST_DIR=/path/to/FolderB

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
  echo "Error: Source directory '$SRC_DIR' does not exist."
  exit 1
fi

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
  echo "Error: Destination directory '$DEST_DIR' does not exist."
  exit 1
fi

# Move files from source to destination
for file in "$SRC_DIR"/*; do
  if [ -f "$file" ]; then
    mv "$file" "$DEST_DIR"
  fi
done
```