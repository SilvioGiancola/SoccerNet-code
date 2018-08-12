# Frame feature extraction

We provide generic code to extract the frame features.



## ResNET

We used ResNET-152 pre-trained on ImageNet available [here](https://drive.google.com/open?id=1sKKUH2Ozawu3epyg3YL6jsQ_f6dPcBJ_) (place the .h5 file into this folder).

- Create and activate feature extraction environment:

`conda env create -f src/feature_extraction/environment.yml`

`source activate Soccer-FeatureExtractor`

- Extract frame features:

`python src/feature_extraction main.py --ResNET`



## C3D

We used C3D pre-trained on Sports1M available [here](https://drive.google.com/open?id=1KxtgwkX_X0sfzJSZFVz8bIIxkufYN5kr) (place the .h5 file into this folder).

- Create and activate feature extraction environment:

`conda env create -f src/feature_extraction/environment.yml`

`source activate Soccer-FeatureExtractor`

- Extract frame features:

`python src/feature_extraction main.py --C3D`



## I3D

- Create and activate feature extraction environment:

`cd src/feature_extraction/i3d-feat-extract/`

`conda env create -f environment.yml`

`source activate i3d-feat-extract`


- Get the original Kinetics-I3D:

Clone Kinetics-I3D: 

`git clone https://github.com/deepmind/kinetics-i3d.git`


- update `$PYTHONPATH`:

`export PYTHONPATH='<you_main_soccernet_github_path>/src/feature_extraction/i3d-feat-extract/kinetics-i3d':$PYTHONPATH`


- Extract frame features:

`python extract_i3d_spatial_features.py <you_main_soccernet_github_path>/data/ <you_main_soccernet_github_path>/data/`

Due to the long computation time, it is recommended to parallelize the code for the I3D feature extraction.
To do so, we used our cluster and the argument `--jobid <jobid>` (jobid from 0 to 499) to specify a single job per game.
An example is provided with the file  `/src/feature_extraction/i3d-feat-extract/extract_I3D_football.sh` using SLURM.



