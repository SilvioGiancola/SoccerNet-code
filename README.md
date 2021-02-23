# SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos

## [DEPRECATED] Please visit https://github.com/SilvioGiancola/SoccerNetv2-DevKit for an updated version of that repository

CVPR'18 Workshop on Computer Vision in Sports

Available at [openaccess.thecvf.com](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf)

```bibtex
@InProceedings{Giancola_2018_CVPR_Workshops,
  author = {Giancola, Silvio and Amine, Mohieddine and Dghaily, Tarek and Ghanem, Bernard},
  title = {SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

Project page: https://silviogiancola.github.io/SoccerNet/

Data available:

- [Frames Features](https://drive.google.com/drive/folders/1qkIeQCGaHg0_CUCHvh3hQFTlq26D20Ts?usp=sharing) (119.9GB)
[[pre-zipped]](https://drive.google.com/file/d/1FDyrfnp8dsF7cd_NyzdTHA-EFIK_TvKK/view?usp=sharing)
- [Labels](https://drive.google.com/drive/folders/1j95bI6G8q434K22wxWRvz2ymA8FF3rei?usp=sharing) (292.9kB)
[[pre-zipped]](https://drive.google.com/file/d/10-Y5yqH8YQ0_lvppWPMSLq6SMayuWT4E/view?usp=sharing)
- [Commentaries](https://drive.google.com/drive/folders/1XD7Kiqw7rsmMn6fYDxN82BdlD_HfkF49?usp=sharing) (26.1MB)
[[pre-zipped]](https://drive.google.com/file/d/1BgPwrHzuz5WDZqmll9K2koP0k0932TNW/view?usp=sharing)
- Videos (224p and HQ): fill the [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13) first (links provided after agreeing with the 
[NDA](https://drive.google.com/file/d/1_e9oZ3rp6hHA2Hm2tjUDMBXYqVrlUKwj/view?usp=sharing)):

### Clone this repository

`git clone https://github.com/SilvioGiancola/SoccerNet-code.git`

### Create the conda environement (Python3)

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

## Read data

### Read data for a single game

`python src/ReadData.py "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"`

### Read commentaries for a single game

`python src/ReadCommentaries.py data france_ligue-1 2016-2017 "Paris SG" "Marseille"`

### Loop and read over Train/Valid/Test

`python src/ReadSplitData.py data src/listgame_Train_300.npy`

### Loop and read over all games

`python src/ReadAllData.py data`

## Source code for data reproducibility

### Features Extraction from videos

See [src/feature_extraction](src/feature_extraction/) for more details.

### Action Classification

See [src/Classification](src/Classification/) for more details.

### Action Detection/Spotting

See [src/Detection](src/Detection/) for more details.

## Getting Started with [Colab](https://colab.research.google.com/notebooks/welcome.ipynb)

It is possible to use Colab to work with SoccerNet on the Google Cloud.
Colab provides a colaborative python environment in the cloud including *unlimited storage* as well as a *free Tesla K80 GPU*.

To us SoccerNet on Colab, please check this [jupyter notebook](https://colab.research.google.com/drive/0B2t5TGieUKOCZlJ4RXVKc1c5UkZ1V2FPbGliSTJqVW9CSjN3).

(Acknowlegments: thanks to [lamia13Alg](https://github.com/lamia13Alg) for sharing her Colab notebook)
