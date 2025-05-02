```bash
#!/bin/bash

find /testbed -type l -print0 | while IFS= read -r -d $'\0' link;
do
  target=$(readlink "$link")
  echo "$target"
done
```