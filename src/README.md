# SoccerNET Release V1

Here is the source code to download and use the SoccerNet dataset


# Download the data

We recommand to use https://github.com/wkentaro/gdown to download large files from google drive.

`pip install gdown` (already in the conda environment)

Please use the following script to download automatically the data:


 - Frames Features:

`./src/SoccerNet_CSV_Downloader.sh data/SoccerNet_V1.1_Features.csv`


 - Labels:

`./src/SoccerNet_CSV_Downloader.sh data/SoccerNet_V1.1_Labels.csv`


 - Commentaries:

`./src/SoccerNet_CSV_Downloader.sh data/SoccerNet_V1.1_Commentaries.csv`


 - Videos (224p) (csv file available after filling this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13)):

`./src/SoccerNet_CSV_Downloader.sh data/SoccerNet_V1.1_Videos.csv` 

 - Videos (HD) (csv file available after filling this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13)):

`./src/SoccerNet_CSV_Downloader.sh data/SoccerNet_V1.1_Videos_HQ.csv`



# Read data

## Read data for a single game
`python src/ReadData.py "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"`


## Read commentaries for a single game
`python src/ReadCommentaries.py data france_ligue-1 2016-2017 "Paris SG" "Marseille"`


## Loop and read over Train/Valid/Test
`python src/ReadSplitData.py data src/listgame_Train_300.npy`


## Loop and read over all games
`python src/ReadAllData.py data`





# Features Extraction from videos

See [src/feature_extraction](feature_extraction/) for more details.

# Action Classification

See [src/Classification](Classification/) for more details.

# Action Detection/Spotting

See [src/Detection](Detection/) for more details.

