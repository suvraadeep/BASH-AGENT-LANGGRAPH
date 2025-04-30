```bash
#!/bin/bash

# Set the source and target directories
src_dir="/testbed"
target_dir="/testbed/dir3/subdir1/subsubdir1/tmp"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Find and copy files with names containing "FooBar"
find "$src_dir" -type f -name "*FooBar*" -exec cp -v {} "$target_dir"/ \;
```