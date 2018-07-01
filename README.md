# SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos



## Create the conda environement (Python3)
`conda env create -f src/environment.yml`

## Ensure the zip files are in the "Data" folder

## Unzip the data
`python src/UnpackData.py --input_dir data/ --output_dir data/`

## Read data for a single game
`python src/ReadData.py "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"`

## Loop and read over all games
`python src/ReadAllData.py data`