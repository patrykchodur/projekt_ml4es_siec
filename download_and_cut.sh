#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Please provide preset name (eg. chinatree)"
	exit -1
fi

mkdir $1
cd $1
../script.sh ../"$1"_links.txt
../cut.sh *.jpg
mkdir cut
mv cut_* cut
mv cut ../"$1"_cut
