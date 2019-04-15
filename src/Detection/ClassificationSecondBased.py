
from __future__ import print_function, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import time


import numpy as np
import matplotlib.pyplot as plt



from tqdm import tqdm
import random

import pandas as pd

import os

import tensorflow as tf
import csv


def main(args):



    # args.model = "Model/BestModel_Augm_60.ckpt"


    print("Loading Testing Data:", args.testing)
    print("args.PCA:", args.PCA)
    print("args.features:", args.features)
    print("args.network:",args.network)
    print("args.VLAD_k:",args.VLAD_k)
    print("args.model:",args.model)
    print("flush!", flush=True)


    from Dataset import dataset
    my_dataset = dataset()
    my_dataset.loadTestingDataset(path_data=args.testing, featureName=args.features, PCA=args.PCA, window_size_sec=args.WindowSize)


    # define Network
    from Network import networkMinutes
    my_network = networkMinutes(my_dataset, args.network, VLAD_k=args.VLAD_k)



    with tf.Session() as sess:
        print("global_variables_initializer")
        sess.run(tf.global_variables_initializer())

        print("restore session")
        saver = tf.train.Saver().restore(sess, args.model)
        
        sess.run(tf.local_variables_initializer())
        sess.run([my_network.reset_metrics_op])

        start_time = time.time()
        total_num_batches=0
        for i in tqdm(range(my_dataset.nb_batch_testing)):
            
            batch_features, batch_labels, key = my_dataset.getTestingBatch(i)
            
            feed_dict={ my_network.input: batch_features, 
                        my_network.labels: batch_labels,
                        my_network.keep_prob: 1.0,
                        my_network.weights: my_dataset.weights }
            sess.run([my_network.loss], feed_dict=feed_dict) # compute loss
            sess.run(my_network.update_metrics_op, feed_dict=feed_dict) # update metrics
            vals_test = sess.run( my_network.metrics_op, feed_dict=feed_dict ) # return metrics
            predictions = sess.run( my_network.predictions, feed_dict=feed_dict ) # return metrics
            # print()
            # print(key)
            # print(os.path.split(key)[0])
            # print(os.path.split(key)[1])
            # print(predictions.shape)
            # print(predictions.dtype)
            mean_error = np.mean(np.abs(predictions-batch_labels),axis=0)
            # print(mean_error, sum(mean_error))
            predictions_name = os.path.join(os.path.split(key)[0], args.output + "_" + os.path.split(key)[1])
            print(predictions_name)
            np.save(predictions_name, predictions )

            total_num_batches +=1

            vals_test["mAP"] = np.mean([vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]])
        
        good_sample = np.sum( np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
        bad_sample = np.sum( vals_test["confusion_matrix"] - np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
        vals_test["accuracies"] =  good_sample / ( bad_sample + good_sample ) 
        vals_test["accuracy"] = np.mean(vals_test["accuracies"])

        print(vals_test["confusion_matrix"])
        # print(('Batch number: %.3f Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}') % (total_num_batches, vals_test["loss"], training_accuracy, vals_test['mAP']))
        print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
        (vals_test["auc_PR"], vals_test["auc_PR_0"], vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]))
        print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(vals_test['loss'], vals_test["accuracy"], vals_test['mAP']))
        print(' Time: {:<8.3} s'.format(time.time()-start_time))
   


        return vals_test





if __name__ == '__main__':
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)



    parser.add_argument('--testing',    required=False,  type=str,   default='/media/giancos/Football/dataset_crop224/listgame_Test_2.npy',  help='the file containg the testing data.')
    parser.add_argument('--features',   required=False, type=str,   default="ResNET",  help='select typeof features')
    parser.add_argument('--model',      required=False, type=str,   default="Model/BestModel.ckpt",  help='select typeof features')
    parser.add_argument('--output',     required=False, type=str,   default="Predictions",  help='select typeof features')
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--PCA',        required=False, action="store_true",        help='use PCA version of the features')
    parser.add_argument('--network',    required=False, type=str,   default="VLAD",     help='Select the type of network (CNN, MAX, AVERAGE, VLAD)')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--VLAD_k',     required=False, type=int,   default=64,     help='number of cluster for slustering method (NetVLAD, NetRVLAD, NetDBOW, NetFV)' )
    parser.add_argument('--WindowSize', required=False, type=int,   default=60,     help='number of cluster for slustering method (NetVLAD, NetRVLAD, NetDBOW, NetFV)' )
    # --
   

    


    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)


    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=numeric_level)
    delattr(args, 'loglevel')

        

        
    args.PCA = True

    if (args.GPU >= 0):
        import os 
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # if(".csv" in args.csv_file and args.jobid >= 0):        
    #     Params = pd.read_csv(args.csv_file).iloc[args.jobid]
    #     [args.features, args.network, args.imbalance, args.VLAD_k] = [Params.features, Params.network, Params.imbalance, Params.VLAD_k]





    start=time.time()
    main(args)
    logging.info('Total Execution Time is {0} seconds'.format(time.time()-start))





