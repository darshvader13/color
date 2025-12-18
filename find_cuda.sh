#!/bin/bash

directory_path='./'
cuda_phrase='device'

# Use find to locate all Python files in the specified directory and its subdirectories
python_files=$(find "$directory_path" -type f -name "*.py")

# Iterate through each Python file and use grep to search for the CUDA phrase
for file_path in $python_files; do
    if grep -q "$cuda_phrase" "$file_path"; then
        echo "Found '$cuda_phrase' in file: $file_path"
    fi
done
