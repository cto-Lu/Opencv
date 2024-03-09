#!/bin/bash

# 比较两个文件夹中的文件，并将不同的文件复制到指定文件夹

folder1="./JPEGImages"
folder2="/home/stoair/yolov8/data/allImages"
output_folder="./differ"

# 比较两个文件夹中的文件，并将不同的文件复制到指定文件夹
diff_files=$(diff <(cd $folder1 && find . -type f | sort) <(cd $folder2 && find . -type f | sort) | grep "^>" | sed 's/^> //')

for file in $diff_files; do
    cp $folder1/$file $output_folder
    cp $folder2/$file $output_folder
done

