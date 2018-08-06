# SoccerNET Release V1

## Data Folder
This folder contains: 

 - SoccerNet_V1.1_Features.csv
 - SoccerNet_V1.1_Labels.csv
 - SoccerNet_V1.1_Commentaries.csv
 - SoccerNet_V1.1_Videos.csv (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))
 - SoccerNet_V1.1_Videos_HQ.csv (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))


## Data Structure
 - england_epl
   - Season
     - Game
       - 1_C3D.npy (C3D features for the 1st half)
       - 1_I3D.npy (I3D features for the 1st half)
       - 1_ResNet.npy (ResNet features for the 1st half)
       - 1_C3D_PCA512.npy (C3D features reduced with PCA for the 1st half)
       - 1_I3D_PCA512.npy (I3D features reduced with PCA for the 1st half)
       - 1_ResNet_PCA512.npy (ResNet features reduced with PCA for the 1st half)
       - 2_C3D.npy (C3D features for the 2nd half)
       - 2_I3D.npy (I3D features for the 2nd half)
       - 2_ResNet.npy (ResNet features for the 2nd half)
       - 2_C3D_PCA512.npy (C3D features reduced with PCA for the 2nd half)
       - 2_I3D_PCA512.npy (I3D features reduced with PCA for the 2nd half)
       - 2_ResNet_PCA512.npy (ResNet features reduced with PCA for the 2nd half)
       - Labels.json (Labels for the game)
       - 1.mkv (video for the 1st time in low quality (240p)) (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))
       - 2.mkv (video for the 2nd time in low quality (240p)) (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))
       - 1_HD.[mkv|mp4|ts] (video for the 1st time in original quality) (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))
       - 2_HD.[mkv|mp4|ts] (video for the 2nd time in original quality) (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))
       - video.ini (start and stop information for the original video) (please fill this [form](https://goo.gl/forms/HXsBbBw6QFMhLvj13))
 - europe_europa-league
 - europe_eufa-champions-league
 - france_ligue-1
 - germany_bundesliga
 - italy_serie_a
 - spain_laliga