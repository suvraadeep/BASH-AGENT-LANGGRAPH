```bash
#!/bin/bash

# Set the directory and pattern
DIR="/workspace"
PATTERN="*.c"

# Find files larger than 1 Kilobyte
find "$DIR" -type f -name "$PATTERN" -size +1k
```