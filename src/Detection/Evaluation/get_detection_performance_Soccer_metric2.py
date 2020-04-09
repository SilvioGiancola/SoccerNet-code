import argparse
import numpy as np
import pandas as pd

from eval_detection import ANETdetection

def main(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True, check_status=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=True)
    res = anet_detection.evaluate()
    return res

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'detection task which is intended to evaluate the ability '
                   'of  algorithms to temporally localize activities in '
                   'untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('--prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--tiou_thresholds', type=float, default=np.linspace(0.5, 0.95, 10),
                   help='Temporal intersection over union threshold.')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    # csv_file = "Results_Detection_metric2.csv"
    for j, Thresh in enumerate(range(95,0,-5)):
        results = []
        for i, Delta in enumerate(range(60,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results/predictions_Thresh_" + str(Thresh) + "_Delta_" + str(Delta) + "_Center.json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Metric2_Tresh_"+str(Thresh)+"_Center", results)


    for j, Thresh in enumerate(range(95,0,-5)):
        results = []
        for i, Delta in enumerate(range(60,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results/predictions_Thresh_" + str(Thresh) + "_Delta_" + str(Delta) + "_ArgMax.json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Metric2_Tresh_"+str(Thresh)+"_ArgMax", results)




        # print("saving results to csv file") 
        # df =  pd.read_csv(csv_file, index_col=0)
        # df.set_value(DeltaGT,str(WaterShedThresh),res)
        # df.to_csv(csv_file, sep=',', encoding='utf-8')
        # print(DeltaGT, WaterShedThresh, res)
