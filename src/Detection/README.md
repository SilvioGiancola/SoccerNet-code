# Action Spotting

cd into the main folder of the repo `SoccerNet-code`.

This code require a model to be trained ofr classification first (see [src/Classification](../Classification/))

Run the following code to get the best model with ResNET and NetVLAD:

`python src/Detection/ClassificationSecondBased.py --testing data/listgame_Test_100.npy --model Model/ResNET_PCA_VLAD64_model.ckpt --features ResNET --network VLAD`

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

More details on the parameters with 
`python src/Detection/ClassificationSecondBased.py --help`
