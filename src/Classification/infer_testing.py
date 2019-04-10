
from __future__ import print_function, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import time


import numpy as np
import matplotlib.pyplot as plt



from tqdm import tqdm
import random
import os



def main(args):

    



    from Dataset_Minute import dataset_Minute
    my_dataset = dataset_Minute()
    # my_dataset.loadTrainingDataset(args.training, args.features, args.PCA, args.context)
    # my_dataset.loadValidationDataset(args.validation, args.features, args.PCA, args.context)
    my_dataset.loadTestingDataset(path_data=args.testing, featureName=args.features, PCA=args.PCA,)




    # define Network
    from Network import networkMinutes
    # my_network = networkMinutes(my_dataset, args.network)
    my_network = networkMinutes(my_dataset, args.network, VLAD_k=args.VLAD_k)

    # my_network.loadModel(args.model)



    # define Trainer
    from Trainer import Trainer
    my_trainer = Trainer(my_network, my_dataset)
    # my_trainer.train(epochs=20000, learning_rate=0.001, , args.tflog)
    
    import csv
    with open('/home/giancos/Dropbox/Applicazioni/ShareLaTeX/CVPR18_Football/img/weight/Weights_Testing_mAP.csv', 'w', newline='') as mAP_file:
        with open('/home/giancos/Dropbox/Applicazioni/ShareLaTeX/CVPR18_Football/img/weight/Weights_Testing_Acc.csv', 'w', newline='') as Acc_file:
    # with open('Weights_Testing_mAP.csv', 'w', newline='') as mAP_file:
    #     with open('Weights_Testing_Acc.csv', 'w', newline='') as Acc_file:

            mAP_csv = csv.writer(mAP_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            acc_csv = csv.writer(Acc_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            mAP_csv.writerow(["mAP", "Logits", "Labels"])
            acc_csv.writerow(["Acc", "Logits", "Labels"])

            models = [s for s in os.listdir(args.ckptpath) if ".ckpt.meta" in s]


            listparam = {
                        "W=1": ["No","No","No","No"], 
                        "W=$\%$": ["large1","logit","large1","label",],
                        "W=$\%^2$": ["large2","logit","large2","label",],
                        "W=$1/\%$": ["small1","logit","small1","label",],
                        "W=$1/\%^2$": ["small2","logit","small2","label",],
                        }

                        
            listparam = {
                        "Nothing": ["Nothing","No","No","No"], 
                        "W=$\%$": ["large1","logit","large1","label",],
                        "W=$\%^2$": ["large2","logit","large2","label",],
                        "W=$1/\%$": ["small1","logit","small1","label",],
                        "W=$1/\%^2$": ["small2","logit","small2","label",],
                        }

            # NO / NO
            for w_str in listparam:
                # print(w_str)
                listparam[w_str][0]

                vals_mAP_logits = vals_Acc_logits = cnt = 0
                for model in [s for s in models if (listparam[w_str][0] in s)]:
                    model = os.path.join(args.ckptpath, model)
                    vals_model = my_trainer.test(model[:-5])
                    vals_mAP_logits += vals_model["mAP"]*100.0
                    vals_Acc_logits += vals_model["Accuracy"]*100.0
                    cnt+=1  
                vals_mAP_logits /= cnt
                vals_Acc_logits /= cnt


                # vals_mAP_labels = vals_Acc_labels = cnt = 0
                # for model in [s for s in models if (listparam[w_str][2] in s and listparam[w_str][3] in s)]:
                #     model = os.path.join(args.ckptpath, model)
                #     vals_model = my_trainer.test(model[:-5])
                #     vals_mAP_labels += vals_model["mAP"]*100.0
                #     vals_Acc_labels += vals_model["Accuracy"]*100.0
                #     cnt+=1  
                # vals_mAP_labels /= cnt
                # vals_Acc_labels /= cnt
                
                mAP_csv.writerow([w_str, "${0:.2f}\% $".format(vals_mAP_logits), "${0:.2f}\% $".format(vals_mAP_labels),])
                acc_csv.writerow([w_str, "${0:.2f}\% $".format(vals_Acc_logits), "${0:.2f}\% $".format(vals_Acc_labels),])


                # vals_mAP = vals_Acc = cnt = 0
                # for model in os.path.join(args.ckptpath, [s for s in models if (listparam[w_str][0] in s and listparam[w_str][1] in s)]):
                #     vals_model = my_trainer.test(model[:-5])
                #     vals_mAP += vals_model["mAP"]/100.0
                #     vals_Acc += vals_model["Accuracy"]/100.0
                #     cnt+=1  
                # vals_mAP /= cnt
                # vals_Acc /= cnt
                


            # # Large1 / Logits
            # vals_mAP = vals_Acc = cnt = 0
            # for model in os.path.join(args.ckptpath, [s for s in models if ("large1" in s and "logit" in s)]):
            #     vals_model = my_trainer.test(model[:-5])
            #     vals_mAP += vals_model["mAP"]/100.0
            #     vals_Acc += vals_model["Accuracy"]/100.0
            #     cnt+=1  
            # vals_mAP /= cnt
            # vals_Acc /= cnt


            # mAP_csv.writerow(["W=1", "${0:.2f}\% $".format(vals_mAP), "${0:.2f}\% $".format(vals_mAP),])
            # mAP_csv.writerow(["W=1", "${0:.2f}\% $".format(vals_mAP), "${0:.2f}\% $".format(vals_mAP),])

            # vals_logits = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("large1" in s and "logit" in s)][-1][:-5]))
            # vals_labels = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("large1" in s and "label" in s)][-1][:-5]))    
            # mAP_csv.writerow(["W=$\%$", "${0:.2f}\% $".format(vals_logits["mAP"]*100 ), "${0:.2f}\% $".format(vals_labels["mAP"]*100),])
            # acc_csv.writerow(["W=$\%$", "${0:.2f}\% $".format(vals_logits["Accuracy"]*100 ), "${0:.2f}\% $".format(vals_labels["Accuracy"]*100),])

            # vals_logits = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("large2" in s and "logit" in s)][-1][:-5]))
            # vals_labels = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("large2" in s and "label" in s)][-1][:-5]))    
            # mAP_csv.writerow(["W=$\%^2$", "${0:.2f}\% $".format(vals_logits["mAP"]*100 ), "${0:.2f}\% $".format(vals_labels["mAP"]*100 ),])
            # acc_csv.writerow(["W=$\%^2$", "${0:.2f}\% $".format(vals_logits["Accuracy"]*100 ), "${0:.2f}\% $".format(vals_labels["Accuracy"]*100 ),])

            # vals_logits = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("small1" in s and "logit" in s)][-1][:-5]))
            # vals_labels = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("small1" in s and "label" in s)][-1][:-5]))    
            # mAP_csv.writerow(["W=1/$\%$", "${0:.2f}\% $".format(vals_logits["mAP"]*100 ), "${0:.2f}\% $".format(vals_labels["mAP"]*100 ),])
            # acc_csv.writerow(["W=1/$\%$", "${0:.2f}\% $".format(vals_logits["Accuracy"]*100 ), "${0:.2f}\% $".format(vals_labels["Accuracy"]*100 ),])

            # vals_logits = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("small2" in s and "logit" in s)][-1][:-5]))
            # vals_labels = my_trainer.test(os.path.join(args.ckptpath, [s for s in models if ("small2" in s and "label" in s)][-1][:-5]))    
            # mAP_csv.writerow(["W=1/$\%^2$", "${0:.2f}\% $".format(vals_logits["mAP"]*100 ), "${0:.2f}\% $".format(vals_labels["mAP"]*100 ),])
            # acc_csv.writerow(["W=1/$\%^2$", "${0:.2f}\% $".format(vals_logits["Accuracy"]*100 ), "${0:.2f}\% $".format(vals_labels["Accuracy"]*100 ),])
           


if __name__ == '__main__':
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)


    parser.add_argument('-testing', '--testing', required=False, type=str,
                      help='the file containg the training data.', default="/media/giancos/Football/dataset_crop224/listgame_Test_100.npy")   

    parser.add_argument('--GPU',        help='ID of the GPU to use' ,   required=False, type=int,   default=-1)
    parser.add_argument('--features',   help='select typeof features',  required=True,  type=str,   default="C3D")
    parser.add_argument('--PCA',        help='use PCA version of the features', required=False, action="store_true")
    parser.add_argument('--ckptpath',      help='path of a pretrained model',  required=False, type=str,   default="")
    parser.add_argument('--network',    required=False, type=str,   default="",     help='Select the type of network (CNN, MAX, AVERAGE, VLAD)')
    parser.add_argument('--VLAD_k',     required=False, type=int,   default=64,        help='maximum number of epochs' )
    parser.add_argument('--loglevel',   default='INFO', help='logging level')


    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    level=numeric_level)
    delattr(args, 'loglevel')

    if (args.GPU >= 0):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    
    start=time.time()
    main(args)
    logging.info('Total Execution Time is {0}'.format(time.time()-start)+ ' seconds')
