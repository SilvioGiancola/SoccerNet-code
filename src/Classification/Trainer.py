

import time
from datetime import date
from tqdm import *
import tensorflow as tf
import numpy as np
import os
import math
import heapq
from operator import itemgetter



class Trainer():
    def __init__(self, network, dataset):
        self.network = network
        self.dataset = dataset


    def get_Accuracy_From_Confusion_Matrix(self,confusion_matrix):
        good_sample = np.sum( np.multiply(confusion_matrix, np.identity(4)), axis=0)
        bad_sample = np.sum( confusion_matrix - np.multiply(confusion_matrix, np.identity(4)), axis=0)
        accuracy = 1 - np.mean( bad_sample / ( bad_sample + good_sample ) )
        return accuracy


    def infer(self, sess, indexes):

        nb_data = len(indexes)
        size_batch = 10000
        nb_batch = math.ceil(nb_data/size_batch)
        for i in range(nb_batch):
            
            init = i*size_batch
            until = min((i+1)*size_batch, nb_data)
            infer_sample_indices = indexes[init:until]


            infer_sample_features = []
            infer_sample_labels   = []

            for index in infer_sample_indices:
                infer_sample_features.append( self.dataset.training_features[index[0]][index[1]])
                infer_sample_labels.append(  self.dataset.training_Labels_onehot[index[0]][index[1]])

            infer_sample_features = np.array(infer_sample_features)
            infer_sample_labels = np.array(infer_sample_labels)
            feed_dict={ self.network.input: infer_sample_features, 
                        self.network.labels: infer_sample_labels,
                        self.network.keep_prob: 1.0,
                        self.network.weights: self.dataset.weights, }
            predictions = sess.run(self.network.predictions, feed_dict=feed_dict) # get prediction

            for index in tqdm(range(len(predictions))):
                prediction = predictions[index][:]
                label = infer_sample_labels[index][:]

                if (label[0] == 1): infer_sample_indices[index][2] = prediction[0]
                if (label[1] == 1): infer_sample_indices[index][2] = prediction[1]
                if (label[2] == 1): infer_sample_indices[index][2] = prediction[2]
                if (label[3] == 1): infer_sample_indices[index][2] = prediction[3]
                       



    def train(self, epochs=1, learning_rate=0.001, tflog="logs"):
        self.tflog = tflog


        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            # define writer
            output_prefix = self.dataset.featureName + "_" + ("PCA_" if self.dataset.PCA else "") + self.network.network_type + str(self.network.VLAD_k) + "_" + self.dataset.imbalance + "_" + str(date.today().isoformat() + time.strftime('_%H-%M-%S'))

            saver =  tf.train.Saver()
            if (not os.path.exists(self.tflog)):
                os.makedirs(self.tflog)                
            self.train_writer = tf.summary.FileWriter(os.path.join(self.tflog, output_prefix + "_training"), sess.graph)
            self.valid_writer = tf.summary.FileWriter(os.path.join(self.tflog, output_prefix + "_validation"), sess.graph)
            self.test_writer  = tf.summary.FileWriter(os.path.join(self.tflog, output_prefix + "_testing"), sess.graph)

            best_validation_mAP = 0
            best_validation_accuracy = 0
            best_validation_loss = 9e9
            best_epoch = 0
            cnt_since_best_epoch = 0

            # Training cycle
            for epoch in range(epochs):
                start_time_epoch = time.time()
                print("\n\n\n")
                print('Epoch {:>2}, Football:  '.format(epoch + 1))
                print("weights:", self.dataset.weights)








                # set prediction to value
                if ("HNMsmall" in self.dataset.imbalance):
                    sess.run(tf.local_variables_initializer())
                    sess.run([self.network.reset_metrics_op])

                    print("HNM - Infering for complete dataset")
                    start_time = time.time()
                    total_num_batches=0
                    # for i in  : self.train_sample_indices.append(i)


                    # print("maxgoal:", max(l[2] for l in self.dataset.training_indices_goal ), "mingoal:", heapq.nsmallest(500, self.dataset.training_indices_goal, key=itemgetter(2))[400][2])
                    self.infer(sess, self.dataset.training_indices_goal)
                    # print("maxgoal:", max(l[2] for l in self.dataset.training_indices_goal ), "mingoal:", heapq.nsmallest(500, self.dataset.training_indices_goal, key=itemgetter(2))[400][2])

                    # print("maxsubs:", max(l[2] for l in self.dataset.training_indices_subs ))
                    self.infer(sess, self.dataset.training_indices_subs)
                    # print("maxsubs:", max(l[2] for l in self.dataset.training_indices_subs ))

                    # print("maxcard:", max(l[2] for l in self.dataset.training_indices_card ))
                    self.infer(sess, self.dataset.training_indices_card)
                    # print("maxcard:", max(l[2] for l in self.dataset.training_indices_card ))

                    # print("maxback:", max(l[2] for l in self.dataset.training_indices_back ))
                    self.infer(sess, self.dataset.training_indices_back)
                    # print("maxback:", max(l[2] for l in self.dataset.training_indices_back ))











                self.dataset.prepareNewEpoch()



                # Training
                print("\n")
                print('Training')

                sess.run([self.network.reset_metrics_op])
                sess.run(tf.local_variables_initializer())
               
                start_time = time.time()
                # total_num_batches = 0
                for total_num_batches in tqdm(range(self.dataset.nb_batch_training)):

                    batch_features, batch_labels, batch_indices = self.dataset.getTrainingBatch(total_num_batches)                    
                    feed_dict={ self.network.input: batch_features, 
                                self.network.labels: batch_labels,
                                self.network.keep_prob: 0.6,
                                self.network.learning_rate: learning_rate,
                                self.network.weights: self.dataset.weights,}

                    # xent1,xent2,xent3,xent4 = sess.run([self.network.cross_entropy_test1,self.network.cross_entropy_test2,self.network.cross_entropy_test3,self.network.cross_entropy_test4,], feed_dict=feed_dict) # get Xent
                    # print("xent1:", xent1)
                    # print("xent2:", xent2)
                    # print("xent3:", xent3)
                    # print("xent4:", xent4)
                    # print("weighted_predictions_2:", weighted_predictions_2)
                    # print("label_2 - weighted_predictions_2:", label_2 - weighted_predictions_2)
                    # print("logits_0:", logits_0)
                    for sub_epoch in range(self.dataset.nb_epoch_per_batch): 
                        if(self.dataset.nb_epoch_per_batch > 1): print("  -- sub_epoch:", sub_epoch)
                        sess.run(self.network.optimizer, feed_dict=feed_dict) # optimize


                        sess.run(self.network.update_metrics_op, feed_dict=feed_dict) # update metrics
                        vals_train = sess.run( self.network.metrics_op, feed_dict=feed_dict ) # return metrics
                        total_num_batches += 1

                        good_sample = np.sum( np.multiply(vals_train["confusion_matrix"], np.identity(4)), axis=0)
                        bad_sample = np.sum( vals_train["confusion_matrix"] - np.multiply(vals_train["confusion_matrix"], np.identity(4)), axis=0)
                        vals_train["accuracies"] =  good_sample / ( bad_sample + good_sample ) 
                        vals_train["accuracy"] = np.mean(vals_train["accuracies"])
                        vals_train["mAP"]  = np.mean([vals_train["auc_PR_1"], vals_train["auc_PR_2"], vals_train["auc_PR_3"]])

                        print(('Batch number: %.3f Loss: %.3f Accuracy: %.3f mAP: %.3f') % (total_num_batches, vals_train["loss"], vals_train["accuracy"], vals_train['mAP']))
                        print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
                        (vals_train["auc_PR"], vals_train["auc_PR_0"], vals_train["auc_PR_1"], vals_train["auc_PR_2"], vals_train["auc_PR_3"]))
                        



                print(vals_train["confusion_matrix"])
                    # self.dataset.updateResults(predictions, batch_labels, batch_indices)

                print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(vals_train['loss'], vals_train["accuracy"], vals_train['mAP']))
                print(' Time: {:<8.3} s'.format(time.time()-start_time), flush=True)
                
                summaries = [
                    tf.Summary.Value(tag="learning_rate",       simple_value=learning_rate),
                    tf.Summary.Value(tag="loss",                simple_value=vals_train['loss']),
                    tf.Summary.Value(tag="accuracy/average",    simple_value=vals_train["accuracy"]),
                    tf.Summary.Value(tag="accuracy/0_background",   simple_value=vals_train["accuracies"][0]),
                    tf.Summary.Value(tag="accuracy/1_cards",    simple_value=vals_train["accuracies"][1]),
                    tf.Summary.Value(tag="accuracy/2_subs",     simple_value=vals_train["accuracies"][2]),
                    tf.Summary.Value(tag="accuracy/3_goals",    simple_value=vals_train["accuracies"][3]),
                    tf.Summary.Value(tag="AP/mean",             simple_value=vals_train["mAP"]),
                    tf.Summary.Value(tag="AP/0_background",     simple_value=vals_train["auc_PR_0"]),
                    tf.Summary.Value(tag="AP/1_cards",          simple_value=vals_train["auc_PR_1"]),
                    tf.Summary.Value(tag="AP/2_subs",           simple_value=vals_train["auc_PR_2"]),
                    tf.Summary.Value(tag="AP/3_goals",          simple_value=vals_train["auc_PR_3"]),
                ]
                self.train_writer.add_summary(tf.Summary(value=summaries), epoch)










                # Run Multiple Validation, to check the metrics are constant 
                for i_val in range(2):

                    print("\n")
                    print('Validation', i_val)

                    vals_valid = self.validate(sess)


                summaries = [
                    tf.Summary.Value(tag="learning_rate",       simple_value=learning_rate),
                    tf.Summary.Value(tag="loss",                simple_value=vals_valid['loss']),
                    tf.Summary.Value(tag="accuracy/average",    simple_value=vals_valid["accuracy"]),
                    tf.Summary.Value(tag="accuracy/0_background",   simple_value=vals_valid["accuracies"][0]),
                    tf.Summary.Value(tag="accuracy/1_cards",    simple_value=vals_valid["accuracies"][1]),
                    tf.Summary.Value(tag="accuracy/2_subs",     simple_value=vals_valid["accuracies"][2]),
                    tf.Summary.Value(tag="accuracy/3_goals",    simple_value=vals_valid["accuracies"][3]),
                    tf.Summary.Value(tag="AP/mean",             simple_value=vals_valid["mAP"]),
                    tf.Summary.Value(tag="AP/0_background",     simple_value=vals_valid["auc_PR_0"]),
                    tf.Summary.Value(tag="AP/1_cards",          simple_value=vals_valid["auc_PR_1"]),
                    tf.Summary.Value(tag="AP/2_subs",           simple_value=vals_valid["auc_PR_2"]),
                    tf.Summary.Value(tag="AP/3_goals",          simple_value=vals_valid["auc_PR_3"]),
                ]
                self.valid_writer.add_summary(tf.Summary(value=summaries), epoch)







                # Look for best model
                print("\n")
                print("validation_mAP: " + str(vals_valid['mAP']))
                print("best_validation_mAP: " + str(best_validation_mAP))
                print("validation_mAP > best_validation_mAP ?: " + str(vals_valid['mAP'] > best_validation_mAP))
                print("cnt_since_best_epoch currently: " + str(cnt_since_best_epoch))
                print("elapsed time for this epoch: " + str(time.time() - start_time_epoch))
                if(vals_valid['mAP'] > best_validation_mAP):
                    best_validation_mAP = vals_valid['mAP']
                    best_validation_accuracy = vals_valid["accuracy"]
                    best_validation_loss = vals_valid['loss']
                    best_epoch = epoch
                    cnt_since_best_epoch = 0
                    saver.save(sess, os.path.join(self.tflog, output_prefix + "_model.ckpt"))
                else:
                    cnt_since_best_epoch += 1


                if (cnt_since_best_epoch > 10 ) and (learning_rate > 0.0001):
                    print("reducing LR after plateau since " + str(cnt_since_best_epoch) + " epochs without improvements")
                    learning_rate /= 10

                    # best_validation_mAP = vals_valid['mAP']
                    # best_validation_accuracy = vals_valid["accuracy"]
                    # best_validation_loss = vals_valid['loss']
                    # best_epoch = epoch
                    cnt_since_best_epoch = 0
                    saver.restore(sess, os.path.join(self.tflog, output_prefix + "_model.ckpt"))

                    # saver.save(sess, os.path.join(self.tflog, output_prefix + "_model.ckpt"))

                elif (cnt_since_best_epoch > 30 ):
                    print("stopping after plateau since " + str(cnt_since_best_epoch) + " epochs without improvements")
                    break;



            self.train_writer.close()
            self.valid_writer.close()
            print("stopping after " + str(epoch) + " epochs maximum training reached")






            print("\n")
            print('Testing')
            saver.restore(sess, os.path.join(self.tflog, output_prefix + "_model.ckpt"))


            vals_test = self.test(sess)




            # good_sample = np.sum( np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
            # bad_sample = np.sum( vals_test["confusion_matrix"] - np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
            # testing_accuracies =  good_sample / ( bad_sample + good_sample ) 
            # testing_accuracy = np.mean(training_accuracies)
            # vals_test["mAP"]  = np.mean([vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]])



            # print(('Testing: Loss: %.3f Accuracy: %.3f mAP: %.3f') % (vals_test["loss"], testing_accuracy, vals_test['mAP']))
            # print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
            # (vals_test["auc_PR"], vals_test["auc_PR_0"], vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]))
                        

            # print(vals_test["confusion_matrix"])
                # self.dataset.updateResults(predictions, batch_labels, batch_indices)

            # print(' Time: {:<8.3} s'.format(time.time()-start_time))


            summaries = [
                tf.Summary.Value(tag="learning_rate",       simple_value=learning_rate),
                tf.Summary.Value(tag="loss",                simple_value=vals_test['loss']),
                tf.Summary.Value(tag="accuracy/average",    simple_value=vals_test["accuracy"]),
                tf.Summary.Value(tag="accuracy/0_background",   simple_value=vals_test["accuracies"][0]),
                tf.Summary.Value(tag="accuracy/1_cards",    simple_value=vals_test["accuracies"][1]),
                tf.Summary.Value(tag="accuracy/2_subs",     simple_value=vals_test["accuracies"][2]),
                tf.Summary.Value(tag="accuracy/3_goals",    simple_value=vals_test["accuracies"][3]),
                tf.Summary.Value(tag="AP/mean",             simple_value=vals_test["mAP"]),
                tf.Summary.Value(tag="AP/0_background",     simple_value=vals_test["auc_PR_0"]),
                tf.Summary.Value(tag="AP/1_cards",          simple_value=vals_test["auc_PR_1"]),
                tf.Summary.Value(tag="AP/2_subs",           simple_value=vals_test["auc_PR_2"]),
                tf.Summary.Value(tag="AP/3_goals",          simple_value=vals_test["auc_PR_3"]),
            ]
            self.test_writer.add_summary(tf.Summary(value=summaries), epoch)
            self.test_writer.close()

        return vals_train, vals_valid, vals_test, output_prefix




    def validate(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run([self.network.reset_metrics_op])

        start_time = time.time()
        total_num_batches=0
        for i in tqdm(range(self.dataset.nb_batch_validation)):
            
            batch_features, batch_labels = self.dataset.getValidationBatch(i)
            
            feed_dict={ self.network.input: batch_features, 
                        self.network.labels: batch_labels,
                        self.network.keep_prob: 1.0,
                        self.network.weights: self.dataset.weights, }
            # sess.run([self.network.loss], feed_dict=feed_dict) # compute loss
            sess.run(self.network.update_metrics_op, feed_dict=feed_dict) # update metrics
            vals_valid = sess.run( self.network.metrics_op, feed_dict=feed_dict ) # return metrics

            total_num_batches +=1

            vals_valid["mAP"] = np.mean([vals_valid["auc_PR_1"], vals_valid["auc_PR_2"], vals_valid["auc_PR_3"]])
        
        good_sample = np.sum( np.multiply(vals_valid["confusion_matrix"], np.identity(4)), axis=0)
        bad_sample = np.sum( vals_valid["confusion_matrix"] - np.multiply(vals_valid["confusion_matrix"], np.identity(4)), axis=0)
        vals_valid["accuracies"] =  good_sample / ( bad_sample + good_sample ) 
        vals_valid["accuracy"] = np.mean(vals_valid["accuracies"])

        print(vals_valid["confusion_matrix"])
        print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
        (vals_valid["auc_PR"], vals_valid["auc_PR_0"], vals_valid["auc_PR_1"], vals_valid["auc_PR_2"], vals_valid["auc_PR_3"]))
        print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(vals_valid['loss'], vals_valid["accuracy"], vals_valid['mAP']))
        print(' Time: {:<8.3} s'.format(time.time()-start_time))

        return vals_valid








    def test(self, sess):


        
        sess.run(tf.local_variables_initializer())
        sess.run([self.network.reset_metrics_op])

        start_time = time.time()
        total_num_batches=0
        for i in tqdm(range(self.dataset.nb_batch_testing)):
            
            batch_features, batch_labels = self.dataset.getTestingBatch(i)
            
            feed_dict={ self.network.input: batch_features, 
                        self.network.labels: batch_labels,
                        self.network.keep_prob: 1.0,
                        self.network.weights: self.dataset.weights }
            sess.run([self.network.loss], feed_dict=feed_dict) # compute loss
            sess.run(self.network.update_metrics_op, feed_dict=feed_dict) # update metrics
            vals_test = sess.run( self.network.metrics_op, feed_dict=feed_dict ) # return metrics
            # sess.run(self.network.update_metrics_op, feed_dict=feed_dict)
            # vals_test = sess.run( self.network.metrics_op, feed_dict=feed_dict )

            total_num_batches +=1

            vals_test["mAP"] = np.mean([vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]])
        # print(('Batch number: %.3f loss: %.3f ') % (total_num_batches, vals_test['loss']))
        # print(('auc: %.3f ') % (vals_test["auc_PR"]))
        # print(('auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f ') %
        # (vals_test["auc_PR_0"], vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]))
        
        good_sample = np.sum( np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
        bad_sample = np.sum( vals_test["confusion_matrix"] - np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
        vals_test["accuracies"] =  good_sample / ( bad_sample + good_sample ) 
        vals_test["accuracy"] = np.mean(vals_test["accuracies"])
        # validation_accuracies =  good_sample / ( bad_sample + good_sample ) 
        # vals_test["Accuracy"] = np.mean(validation_accuracies)

        print(vals_test["confusion_matrix"])
        # print(('Batch number: %.3f Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}') % (total_num_batches, vals_test["loss"], training_accuracy, vals_test['mAP']))
        print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
        (vals_test["auc_PR"], vals_test["auc_PR_0"], vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]))
        print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(vals_test['loss'], vals_test["accuracy"], vals_test['mAP']))
        print(' Time: {:<8.3} s'.format(time.time()-start_time))
   


        return vals_test




    # def test(self, model):

    #     saver = tf.train.Saver()

    #     with tf.Session() as sess:
    #         saver.restore(sess, model)

    #         print("Model restored")
            
    #         print("\n")
    #         print('Testing')
    #         sess.run(tf.local_variables_initializer())
    #         sess.run([self.network.reset_metrics_op])

    #         start_time = time.time()
    #         total_num_batches=0
    #         for i in tqdm(range(self.dataset.nb_batch_testing)):
                
    #             batch_features, batch_labels = self.dataset.getTestingBatch(i)
                
    #             feed_dict={ self.network.input: batch_features, 
    #                         self.network.labels: batch_labels,
    #                         self.network.keep_prob: 1.0,
    #                         self.network.weights: self.dataset.weights }
    #             sess.run([self.network.loss], feed_dict=feed_dict) # compute loss
    #             sess.run(self.network.update_metrics_op, feed_dict=feed_dict) # update metrics
    #             vals_test = sess.run( self.network.metrics_op, feed_dict=feed_dict ) # return metrics
    #             # sess.run(self.network.update_metrics_op, feed_dict=feed_dict)
    #             # vals_test = sess.run( self.network.metrics_op, feed_dict=feed_dict )

    #             total_num_batches +=1

    #             vals_test["mAP"] = np.mean([vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]])
    #         # print(('Batch number: %.3f loss: %.3f ') % (total_num_batches, vals_test['loss']))
    #         # print(('auc: %.3f ') % (vals_test["auc_PR"]))
    #         # print(('auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f ') %
    #         # (vals_test["auc_PR_0"], vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]))
            
    #         good_sample = np.sum( np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
    #         bad_sample = np.sum( vals_test["confusion_matrix"] - np.multiply(vals_test["confusion_matrix"], np.identity(4)), axis=0)
    #         validation_accuracies =  good_sample / ( bad_sample + good_sample ) 
    #         vals_test["Accuracy"] = np.mean(validation_accuracies)

    #         print(vals_test["confusion_matrix"])
    #         # print(('Batch number: %.3f Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}') % (total_num_batches, vals_test["loss"], training_accuracy, vals_test['mAP']))
    #         print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
    #         (vals_test["auc_PR"], vals_test["auc_PR_0"], vals_test["auc_PR_1"], vals_test["auc_PR_2"], vals_test["auc_PR_3"]))
    #         print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(vals_test['loss'], vals_test["Accuracy"], vals_test['mAP']))
    #         print(' Time: {:<8.3} s'.format(time.time()-start_time))
   


    #     return vals_test






