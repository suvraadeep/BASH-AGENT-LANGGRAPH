
#!/bin/bash

# Search for regular files in the testbed directory tree that were last modified more than 7 days ago
find testbed -type f -mtime +7 -exec gzip {} \;
