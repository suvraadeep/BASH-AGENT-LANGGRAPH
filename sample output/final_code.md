#!/bin/bash

# Set the path to the testbed directory tree
TESTBED_DIR="/path/to/testbed"

# Set the compression format
COMPRESSION_FORMAT="gzip"

# Set the file extensions or patterns to exclude from compression (if any)
EXCLUDE_PATTERNS=()

# Find and compress regular files in the testbed directory tree that were last modified more than 7 days ago
find "$TESTBED_DIR" -type f -mtime +7 -not -name "*(${EXCLUDE_PATTERNS[@]})" -exec gzip -v {} \;
