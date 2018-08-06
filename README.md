# SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos

Project page: https://silviogiancola.github.io/SoccerNet/


Data available:
- [Frames Features](https://drive.google.com/drive/folders/1qkIeQCGaHg0_CUCHvh3hQFTlq26D20Ts?usp=sharing) 
[[pre-zipped]](https://drive.google.com/file/d/1FDyrfnp8dsF7cd_NyzdTHA-EFIK_TvKK/view?usp=sharing) (119.9GB)
- [Labels](https://drive.google.com/drive/folders/1j95bI6G8q434K22wxWRvz2ymA8FF3rei?usp=sharing) 
[[pre-zipped]](https://drive.google.com/file/d/10-Y5yqH8YQ0_lvppWPMSLq6SMayuWT4E/view?usp=sharing) (292.9kB)
- [Commentaries](https://drive.google.com/drive/folders/1XD7Kiqw7rsmMn6fYDxN82BdlD_HfkF49?usp=sharing) 
[[pre-zipped]](https://drive.google.com/file/d/1BgPwrHzuz5WDZqmll9K2koP0k0932TNW/view?usp=sharing) (26.1MB)
- Videos (224p and HQ): fill the [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13) first (links provided after agreeing with the 
[NDA](https://drive.google.com/file/d/1_e9oZ3rp6hHA2Hm2tjUDMBXYqVrlUKwj/view?usp=sharing)):



## Clone this repository
`git clone https://github.com/SilvioGiancola/SoccerNet-code.git`

## Create the conda environement (Python3)
`conda env create -f src/environment.yml`

`source activate SoccerNet`


## Download the data

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




## Read data for a single game
`python src/ReadData.py "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"`


## Read commentaries for a single game
`python src/ReadCommentaries.py data france_ligue-1 2016-2017 "Paris SG" "Marseille"`


## Loop and read over Train/Valid/Test
`python src/ReadSplitData.py data src/listgame_Train_300.npy`


## Loop and read over all games
`python src/ReadAllData.py data`
