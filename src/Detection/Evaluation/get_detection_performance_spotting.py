import argparse
import numpy as np

from eval_detection import ANETdetection

def main(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True, check_status=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=True)
    return anet_detection.evaluate()

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




# #### FOR SINGLE CURVE
   

    
    results = []
    for i, Delta in enumerate(range(90,0,-5)):
        args.verbose = True
        args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
        args.prediction_filename = "../Results_Spot/predictions_NMS_50.json"
        args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
        res = main(**vars(args))

        results.append(res)
        print(results)
    np.save("Spotting_NMS", results)


    
    results = []
    for i, Delta in enumerate(range(90,0,-5)):
        args.verbose = True
        args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
        args.prediction_filename = "../Results_Spot/predictions_Center_50.json"
        args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
        res = main(**vars(args))

        results.append(res)
        print(results)
    np.save("Spotting_Center", results)


    results = []
    for i, Delta in enumerate(range(90,0,-5)):
        args.verbose = True
        args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
        args.prediction_filename = "../Results_Spot/predictions_Argmax_50.json"
        args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
        res = main(**vars(args))

        results.append(res)
        print(results)
    np.save("Spotting_Argmax", results)




### FOR EXHAUSTIVE 
    for WaterShedThresh in range(95,0,-5):

        results = []
        for i, Delta in enumerate(range(90,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results_Spot/predictions_Center_"+str(WaterShedThresh)+".json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Spotting_Center_"+str(WaterShedThresh), results)


        results = []
        for i, Delta in enumerate(range(90,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results_Spot/predictions_Argmax_"+str(WaterShedThresh)+".json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Spotting_Argmax_"+str(WaterShedThresh), results)

        results = []
        for i, Delta in enumerate(range(90,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results_Spot/predictions_NMS_"+str(WaterShedThresh)+".json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Spotting_NMS_"+str(WaterShedThresh), results)





### FOR DIFFERENT MODELS (WINDOWS)
    for Wind in [60,50,40,30,20,10,5]:
        print("-----------------WINDOW "+str(Wind)+"-----------------")

        results = []
        print("-----------------Center-----------------")
        for i, Delta in enumerate(range(90,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results_Spot/predictions_"+str(Wind)+"_Center_50.json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Spotting_Center_Wind"+str(Wind), results)


        results = []
        print("-----------------Argmax-----------------")
        for i, Delta in enumerate(range(90,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results_Spot/predictions_"+str(Wind)+"_Argmax_50.json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Spotting_Argmax_Wind"+str(Wind), results)

        results = []
        print("-----------------NMS-----------------")
        for i, Delta in enumerate(range(90,0,-5)):
            args.verbose = True
            args.ground_truth_filename = "../Results/labels_Delta_" + str(Delta) + ".json"
            args.prediction_filename = "../Results_Spot/predictions_"+str(Wind)+"_NMS_50.json"
            args.tiou_thresholds = np.linspace(0.00, 0.00, 1)
            res = main(**vars(args))

            results.append(res)
            print(results)
        np.save("Spotting_NMS_Wind"+str(Wind), results)


        args.prediction_filename = "../Results_Spot/predictions_Argmax.json"


