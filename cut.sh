#!/bin/bash

if [ "$#" -lt 1 ]; then
	echo "Wrong number of arguments. Please provide at least one filename"
	exit -1
fi

echo "Number of files: $#"

counter=0
size=224x224

echo ""

for file in "$@"; do
	convert "$file"  -thumbnail "$size"^ \
          -gravity center -extent "$size"  "cut_$file"

	percent=$(($counter * 100 / $#))
	counter=$(($counter + 1))
	echo -e "\033[1A"
	printf "Convert: %s%%" $percent
done

echo -e "\033[1A"
echo "Convert: 100%"

