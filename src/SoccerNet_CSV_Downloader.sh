#!/bin/bash
# SoccerNet_CSV_Downloader.sh
# Usage:
#	./SoccerNet_CSV_Downloader.sh [CSV FILE]


# destructor
control_c() {
	echo -e "\033[0;31mInterupted at data$j\033[0m"
	echo -e "\033[0;31mThe file has been cleaned\033[0m"
	exit
}

# trap ctrl+c when stuck in gdown
trap control_c SIGINT

# parse CSV file
while IFS=',' read -r j i k
do
    echo -e "\n\033[0;32mParsing $i into data$j \033[0m"

    # create directory
    mkdir -p "data$(dirname "$j")"


    # check if file already exist
	if [[ ! -f "data$j" ]]
	then 
    	# download the google ID in the specific folder
		gdown -O "data$j" --id $i
	else
		echo -e "\033[0;33m /!\ data$j already exist /!\ \033[0m"
    fi

done < $1

