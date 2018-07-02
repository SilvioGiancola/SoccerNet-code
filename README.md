# SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos



## Create the conda environement (Python3)
`conda env create -f src/environment.yml`


## Ensure the zip files are in the "Data" folder
- Videos: fill the [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13) first (links provided after agreeing with the 
[NDA](https://drive.google.com/file/d/1_e9oZ3rp6hHA2Hm2tjUDMBXYqVrlUKwj/view?usp=sharing)):
- Frames Features: 
[[Google Drive]](https://drive.google.com/drive/folders/1qkIeQCGaHg0_CUCHvh3hQFTlq26D20Ts?usp=sharing) 
[[pre-zipped]](https://drive.google.com/drive/folders/1qkIeQCGaHg0_CUCHvh3hQFTlq26D20Ts?usp=sharing) (119.9GB)
- Labels: 
[[Google Drive]](https://drive.google.com/drive/folders/1j95bI6G8q434K22wxWRvz2ymA8FF3rei?usp=sharing) 
[[pre-zipped]](https://drive.google.com/file/d/10-Y5yqH8YQ0_lvppWPMSLq6SMayuWT4E/view?usp=sharing) (292.9kB)
- Commentaries: 
[[Google Drive]](https://drive.google.com/drive/folders/1XD7Kiqw7rsmMn6fYDxN82BdlD_HfkF49?usp=sharing) 
[[pre-zipped]](https://drive.google.com/file/d/1BgPwrHzuz5WDZqmll9K2koP0k0932TNW/view?usp=sharing) (26.1MB)
 


We recommand to use https://github.com/circulosmeos/gdown.pl to download large files.

Usage:

`./gdown.pl https://drive.google.com/file/d/10-Y5yqH8YQ0_lvppWPMSLq6SMayuWT4E/view?usp=sharing data/SoccerNet_V1.1_Features.zip`

`./gdown.pl https://drive.google.com/file/d/10-Y5yqH8YQ0_lvppWPMSLq6SMayuWT4E/view?usp=sharing data/SoccerNet_V1.1_Labels.zip`

`./gdown.pl https://drive.google.com/file/d/1BgPwrHzuz5WDZqmll9K2koP0k0932TNW/view?usp=sharing data/SoccerNet_V1.1_Commentaries.zip`

`idem with the video link (available after agreeing the NDA)`




## Unzip the data
This command will create the data structure and unzip the data

`python src/UnpackData.py --input_dir data/ --output_dir data/`


## Read data for a single game
`python src/ReadData.py "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"`


## Read commentaries for a single game
`python src/ReadCommentaries.py data france_ligue-1 2016-2017 "Paris SG" "Marseille"`


## Loop and read over all games
`python src/ReadAllData.py data`
