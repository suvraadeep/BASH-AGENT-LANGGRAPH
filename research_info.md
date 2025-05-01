What is the correct way to check if a file exists in a directory?

I don’t know.



How can we ensure that the source and destination directories exist before attempting to move files?

if [ -d "$src_dir" ] && [ -d "$dst_dir" ]; then mv "$src_dir"/* "$dst_dir/; fi



What is the most efficient way to move multiple files from one directory to another?

cp file1 file2 file3 ... destination_dir/


