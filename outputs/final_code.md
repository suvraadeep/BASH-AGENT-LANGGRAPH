```bash
#!/bin/bash

# Count md5sum of all '*.py' files in /testbed folder with subfolders

directory="/testbed"

if [ ! -d "$directory" ]; then
  echo "Error: Directory $directory does not exist"
  exit 1
fi

for file in $(find "$directory" -name "*.py"); do
  if [ -r "$file" ]; then
    md5sum "$file"
  else
    echo "Warning: Skipping file $file, no read permission"
  fi
done
```