```bash
#!/bin/bash

SRC_DIR=/path/to/folderA
DEST_DIR=/path/to/FolderB
if [ ! -d "$SRC_DIR" ]; then
echo "Error: Source directory '$SRC_DIR' does not exist."
exit 1
fi
if [ ! -d "$DEST_DIR" ]; then
echo "Error: Destination directory '$DEST_DIR' does not exist."
exit 1
fi
for file in "$SRC_DIR"/*; do
if [ -f "$file" ]; then
mv "$file" "$DEST_DIR"
fi
done
```