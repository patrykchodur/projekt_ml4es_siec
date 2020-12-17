#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Wrong number of arguments. Please provide filename"
	exit -1
fi

lines_count=$(wc -l $1 | sed 's/[[:space:]]*\([[:digit:]]*\).*/\1/')

echo "Number of files: $lines_count"

counter=0

echo ""

rm -f log.txt

while read link; do
	wget -t 2 -T 2 "$link" 2>>log.txt
	percent=$(($counter * 100 / $lines_count))
	counter=$(($counter + 1))
	echo -e "\033[1A"
	printf "Download: %s%%" $percent
done < $1

echo -e "\033[1A"
echo "Download: 100%"

