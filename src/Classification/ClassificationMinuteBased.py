
from __future__ import print_function, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import time


import numpy as np
#import matplotlib.pyplot as plt



from tqdm import tqdm
import random

import pandas as pd



def main(args):

    print("Loading Training Data:", args.training)
    print("args.featureName:", args.features)
    print("args.PCA:", args.PCA)
    print("args.features:", args.features)
    print("args.imbalance:", args.imbalance)
    print("args.network:",args.network)
    print("args.LR:",args.LR)
    print("args.VLAD_k:",args.VLAD_k)
    print("args.max_epoch:",args.max_epoch)
    print("flush!", flush=True)



    from Dataset import dataset
    my_dataset = dataset()
    my_dataset.loadTrainingDataset(path_data=args.training, 
                                    featureName=args.features, 
                                    PCA=args.PCA, 
                                    imbalance=args.imbalance, 
                                    batch_size=args.batch_size, 
                                    window_size_sec = args.WindowSize,)
    my_dataset.loadValidationDataset(path_data=args.validation, featureName=args.features, PCA=args.PCA,
                                    window_size_sec = args.WindowSize,)
    my_dataset.loadTestingDataset(path_data=args.testing, featureName=args.features, PCA=args.PCA,
                                    window_size_sec = args.WindowSize,)




    # define Network
    from Network import networkMinutes
    my_network = networkMinutes(my_dataset, args.network, VLAD_k=args.VLAD_k)



    # define Trainer
    from Trainer import Trainer
    my_trainer = Trainer(my_network, my_dataset)
    vals_train, vals_valid, vals_test, model = my_trainer.train(epochs=args.max_epoch, learning_rate=args.LR, tflog=args.tflog)
    #vals_train, vals_valid, vals_test, model = 0,0,0,"pippo"
    if(".csv" in args.csv_file and args.jobid >= 0 and ("BUTTA" not in args.tflog.upper())):   
        print("saving results to csv file")     
        df = pd.read_csv(args.csv_file, index_col=0)
        df.set_value(args.jobid,"train_mAP",vals_train["mAP"])
        df.set_value(args.jobid,"train_Acc",vals_train["accuracy"])
        df.set_value(args.jobid,"valid_mAP",vals_valid["mAP"])
        df.set_value(args.jobid,"valid_Acc",vals_valid["accuracy"])
        df.set_value(args.jobid,"test_mAP",vals_test["mAP"])
        df.set_value(args.jobid,"test_Acc",vals_test["accuracy"])
        print(model)
        df.set_value(args.jobid,"model",model)
        df.to_csv(args.csv_file, sep=',', encoding='utf-8')


if __name__ == '__main__':
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)


    parser.add_argument('--training',   required=True,  type=str,   help='the file containg the training data.')    
    parser.add_argument('--validation', required=True,  type=str,   help='the file containg the validation data.')
    parser.add_argument('--testing',    required=True,  type=str,   help='the file containg the validation data.')
    parser.add_argument('--features',   required=False, type=str,   default="C3D",  help='select typeof features')
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--PCA',        required=False, action="store_true",        help='use PCA version of the features')
    parser.add_argument('--network',    required=False, type=str,   default="",     help='Select the type of network (CNN, MAX, AVERAGE, VLAD)')
    parser.add_argument('--tflog',      required=False, type=str,   default='Model',   help='folder for tensorBoard output')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--imbalance',  required=False, type=str,   default="No",   help='how to cope with imbalance dataset')
    # parser.add_argument('--norm',       required=False, type=str,   default="No",   help='weights normalization over the largest or the smallest class')
    parser.add_argument('--jobid',      required=False, type=int,   default=-1,     help='Jop ID for batch in Skynet' )
    parser.add_argument('--csv_file',   required=False, type=str,   default="",     help='Jop ID for batch in Skynet' )
    parser.add_argument('--batch_size', required=False, type=int,   default=60,     help='Size of the batch in number of halves game' )
    parser.add_argument('--max_epoch',  required=False, type=int,   default=200,    help='maximum number of epochs' )
    parser.add_argument('--VLAD_k',     required=False, type=int,   default=64,     help='number of cluster for slustering method (NetVLAD, NetRVLAD, NetDBOW, NetFV)' )
    parser.add_argument('--LR',         required=False, type=float, default=0.01,   help='Learning Rate' )
    parser.add_argument('--WindowSize', required=False, type=int,   default=60,     help='Size of the Window' )
    # parser.add_argument('--HNM',      required=False, action="store_true",        help='Mode HNM')


    


    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)


    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=numeric_level)
    delattr(args, 'loglevel')

    if (args.GPU >= 0):
        import os 
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
        

        

    if(".csv" in args.csv_file and args.jobid >= 0):
        Params = pd.read_csv(args.csv_file).iloc[args.jobid]
        [args.features, args.network, args.imbalance, args.VLAD_k, args.WindowSize] = [Params.features, Params.network, Params.imbalance, Params.VLAD_k, Params.WindowSize]





    start=time.time()
    main(args)
    logging.info('Total Execution Time is {0} seconds'.format(time.time()-start))





