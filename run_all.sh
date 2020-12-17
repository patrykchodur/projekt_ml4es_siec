#!/bin/bash

function check_if_command_exists {
	if ! [ command -v "$1" &>/dev/null ]; then
		echo "Error: command $1 does not exist"
		exit -1
	fi
}

check_if_command_exists python3
check_if_command_exists wget
check_if_command_exists convert
check_if_command_exists tflite_convert

for tree in fig judastree palm pine; do
	./download_and_cut.sh $tree
done

python3 net.py

tflite_convert  --output_file=model_converted.tflite  --keras_model_file=model.h5

echo "Everything done!"
