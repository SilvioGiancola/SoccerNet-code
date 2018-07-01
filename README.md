# SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos



## Create the conda environement (Python3)
`conda env create -f src/environment.yml`

## Ensure the zip files are in the "Data" folder
- Frames Features (ResNET PCA 512): [TBD]
- Frames Features (All): [TBD]
- Labels: https://drive.google.com/file/d/1em5K3RToGUpCyAnZl6fwzL-Wv5V1o6aC/view?usp=sharing
- Videos (fill the form): https://goo.gl/forms/HXsBbBw6QFMhLvj13

We recommand to use https://github.com/circulosmeos/gdown.pl to download large files.

Usage:

`./gdown.pl https://drive.google.com/file/d/1em5K3RToGUpCyAnZl6fwzL-Wv5V1o6aC/view?usp=sharing labels.zip`

## Unzip the data
This command will create the data structure and unzip the data

`python src/UnpackData.py --input_dir data/ --output_dir data/`

## Read data for a single game
`python src/ReadData.py "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"`

## Loop and read over all games
`python src/ReadAllData.py data`
