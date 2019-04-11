# minute classifier

cd into the main folder of the repo `SoccerNet-code`.

Run the following code to get the best model with ResNET and NetVLAD:

`python src/Classification/ClassificationMinuteBased.py --training data/listgame_Train_300.npy  --validation data/listgame_Valid_100.npy --testing data/listgame_Test_100.npy --PCA --features ResNET --network VLAD`

`--features` can be 
`ResNET`, 
`C3D` or 
`I3D`.

`--network` can be 
`CNN`,
`FC`,
`MAX`,
`AVERAGE`,
`RVLAD`,
`VLAD`,
`SOFTDBOW` or
`NETFV`.