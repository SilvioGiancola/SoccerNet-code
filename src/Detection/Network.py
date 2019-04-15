import tensorflow as tf
import loupe as lp
import numpy as np
import math



class networkMinutes():

    def __init__(self, dataset, network_type="CNN", VLAD_k=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_k
        self.VLAD_gating = VLAD_gating
        self.VLAD_batch_norm = VLAD_batch_norm
        # define placeholder
        self.input = tf.placeholder(tf.float32, shape=(None, dataset.number_frames_in_window, 512), name="x")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.weights = tf.placeholder(tf.float32, shape=(4), name="weights")
        # self.weights_labels = tf.placeholder(tf.float32, shape=(4), name="weights_labels")
        
        with tf.name_scope('network'):

            x = self.input

            #CNN
            if ("CNN" in network_type.upper()):
                print("Using Convolution Neural Network")
                x = tf.contrib.layers.conv2d(x, num_outputs=128, kernel_size=9, stride=1, padding='SAME')
                x = tf.contrib.layers.flatten(x)
            

            elif ("FC" in network_type.upper()):
                print("Using Fully Connected")
                x = tf.contrib.layers.flatten(x)
                x = tf.contrib.layers.fully_connected(x, dataset.number_frames_in_window*512)


            elif ("MAX" in network_type.upper()):
                print("Using MaxPooling")
                x = tf.layers.max_pooling1d(x, pool_size=dataset.number_frames_in_window, strides=1, padding='SAME', name="MaxPooling")
                x = tf.contrib.layers.flatten(x)


            elif ("AVERAGE" in network_type.upper()):
                print("Using AveragePooling")
                x = tf.layers.average_pooling1d(x, pool_size=dataset.number_frames_in_window, strides=1, padding='SAME', name="MaxPooling")
                x = tf.contrib.layers.flatten(x)



            elif ("RVLAD" in network_type.upper()):
                print("Using NetRVLAD")
                NetRVLAD = lp.NetRVLAD(feature_size=512, max_samples=dataset.number_frames_in_window, cluster_size=VLAD_k, 
                         output_dim=512, gating=VLAD_gating, add_batch_norm=VLAD_batch_norm,
                         is_training=True)
                x = tf.reshape(x, [-1, 512])
                x = NetRVLAD.forward(x)

            elif ("VLAD" in network_type.upper()):
                print("Using NetVLAD")
                NetVLAD = lp.NetVLAD(feature_size=512, max_samples=dataset.number_frames_in_window, cluster_size=VLAD_k, 
                         output_dim=512, gating=VLAD_gating, add_batch_norm=VLAD_batch_norm,
                         is_training=True)
                x = tf.reshape(x, [-1, 512])
                x = NetVLAD.forward(x)

            elif ("SOFTDBOW" in network_type.upper()):
                print("Using SOFTDBOW")
                SOFTDBOW = lp.SoftDBoW(feature_size=512, max_samples=dataset.number_frames_in_window, cluster_size=VLAD_k, 
                         output_dim=512, gating=VLAD_gating, add_batch_norm=VLAD_batch_norm,
                         is_training=True)
                x = tf.reshape(x, [-1, 512])
                x = SOFTDBOW.forward(x)

            elif ("NETFV" in network_type.upper()):
                print("Using NETFV")
                NETFV = lp.NetFV(feature_size=512, max_samples=dataset.number_frames_in_window, cluster_size=VLAD_k, 
                         output_dim=512, gating=VLAD_gating, add_batch_norm=VLAD_batch_norm,
                         is_training=True)
                x = tf.reshape(x, [-1, 512])
                x = NETFV.forward(x)

            x = tf.nn.dropout(x, self.keep_prob)
            x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)


        with tf.name_scope('logits'):
            self.logits = tf.identity(x_output, name='logits')
            # self.logits_0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.logits), 0))
            # self.logits_1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.logits), 1))
            # self.logits_2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.logits), 2))
            # self.logits_3 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.logits), 3))
            # self.logits_0 = tf.transpose(tf.transpose(self.logits)[0])
            # self.logits_1 = tf.transpose(tf.transpose(self.logits)[1])
            # self.logits_2 = tf.transpose(tf.transpose(self.logits)[2])
            # self.logits_3 = tf.transpose(tf.transpose(self.logits)[3])
            # self.weighted_logits = self.weights_logits * self.logits
            # self.weighted_logits_0 = tf.transpose(tf.transpose(self.weighted_logits)[0])
            # self.weighted_logits_1 = tf.transpose(tf.transpose(self.weighted_logits)[1])
            # self.weighted_logits_2 = tf.transpose(tf.transpose(self.weighted_logits)[2])
            # self.weighted_logits_3 = tf.transpose(tf.transpose(self.weighted_logits)[3])
        # dataset.updateResults(self.logits)


        # self.weights = tf.constant(list((dataset.size_batch*1.0)/(dataset.count_labels+1.0)))
        # self.logits_weighted = self.class_weights * self.logits 
        # with tf.name_scope('predictions'):
        with tf.name_scope('predictions'):
            self.predictions = tf.nn.sigmoid(self.logits, name='predictions')
            # self.predictions_0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.predictions), 0))
            # self.predictions_1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.predictions), 1))
            # self.predictions_2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.predictions), 2))
            # self.predictions_3 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.predictions), 3))
            self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
            self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
            self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
            self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

            # self.weighted_predictions = tf.nn.sigmoid(self.weighted_logits, name='weighted_predictions')

            # self.weighted_predictions_0 = tf.transpose(tf.transpose(self.weighted_predictions)[0])
            # self.weighted_predictions_1 = tf.transpose(tf.transpose(self.weighted_predictions)[1])
            # self.weighted_predictions_2 = tf.transpose(tf.transpose(self.weighted_predictions)[2])
            # self.weighted_predictions_3 = tf.transpose(tf.transpose(self.weighted_predictions)[3])
            # self.accuracy_tf = tf.contrib.metrics.streaming_accuracy(self.predictions, self.labels)
        # self.predictions_ = tf.argmax(self.logits, 1)
            # tf.summary.histogram("predictions", self.predictions)


        with tf.name_scope('labels'):
            self.labels = tf.placeholder(tf.float32, shape=(None, 4), name="y")
            # self.labels_0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 0))
            # self.labels_1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 1))
            # self.labels_2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 2))
            # self.labels_3 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 3))
            self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
            self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
            self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
            self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

            # self.weighted_labels = self.weights_labels*self.labels
            # self.weighted_labels_0 = tf.transpose(tf.transpose(self.weighted_labels)[0])
            # self.weighted_labels_1 = tf.transpose(tf.transpose(self.weighted_labels)[1])
            # self.weighted_labels_2 = tf.transpose(tf.transpose(self.weighted_labels)[2])
            # self.weighted_labels_3 = tf.transpose(tf.transpose(self.weighted_labels)[3])



        # with tf.name_scope('weights'):
        #     # self.labels = tf.placeholder(tf.float32, shape=(None, 4), name="y")
        #     # self.weights_logits_0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_logits), 0))
        #     # self.weights_logits_1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_logits), 1))
        #     # self.weights_logits_2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_logits), 2))
        #     # self.weights_logits_3 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_logits), 3))
        #     self.weights_logits_0 = tf.transpose(tf.transpose(self.se)[0])
        #     self.weights_logits_1 = tf.transpose(tf.transpose(self.weights_logits)[1])
        #     self.weights_logits_2 = tf.transpose(tf.transpose(self.weights_logits)[2])
        #     self.weights_logits_3 = tf.transpose(tf.transpose(self.weights_logits)[3])

        #     # self.weights_labels_0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_labels), 0))
        #     # self.weights_labels_1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_labels), 1))
        #     # self.weights_labels_2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_labels), 2))
        #     # self.weights_labels_3 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.weights_labels), 3))
        #     self.weights_labels_0 = tf.transpose(tf.transpose(self.weights_labels)[0])
        #     self.weights_labels_1 = tf.transpose(tf.transpose(self.weights_labels)[1])
        #     self.weights_labels_2 = tf.transpose(tf.transpose(self.weights_labels)[2])
        #     self.weights_labels_3 = tf.transpose(tf.transpose(self.weights_labels)[3])
        # #     self.labels_1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 1))
        # #     self.labels_2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 2))
        # #     self.labels_3 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.labels), 3))




        with tf.name_scope('cost'):
            # https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow
            # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.weights_logits*self.logits, labels=self.weights_labels*self.labels) 
            # self.cross_entropy = self.weights_labels_0
            # self.cross_entropy_0 = -self.weighted_labels_0 * tf.log(self.weighted_predictions_0)/4
            # self.cross_entropy_1 = -self.weighted_labels_1 * tf.log(self.weighted_predictions_1)/4
            # self.cross_entropy_2 = -self.weighted_labels_2 * tf.log(self.weighted_predictions_2)/4
            # self.cross_entropy_3 = -self.weighted_labels_3 * tf.log(self.weighted_predictions_3)/4
            # self.cross_entropy = self.cross_entropy_0 + self.cross_entropy_1 + self.cross_entropy_2 + self.cross_entropy_3
            # self.cross_entropy = - tf.reduce_sum( ( (self.weighted_labels*tf.log(self.weighted_predictions + 1e-9)) + ((1-self.weighted_labels) * tf.log(1 - self.weighted_predictions + 1e-9)) )  , name='xentropy' )   
            self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(logits = self.logits, 
                                                                            targets = self.labels,
                                                                            pos_weight = self.weights)
            self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

            # self.cross_entropy_test1 = tf.reduce_sum(
            #     tf.nn.weighted_cross_entropy_with_logits(logits = self.logits, 
            #                                             targets = self.weights_labels * self.labels,
            #                                             pos_weight = self.weights_logits))
            # self.cross_entropy_test2 = tf.reduce_sum(
            #     tf.nn.weighted_cross_entropy_with_logits(logits = self.logits, 
            #                                             targets = self.labels,
            #                                             pos_weight = self.weights_labels))           
            # self.cross_entropy_test3 = tf.reduce_sum(
            #     tf.nn.weighted_cross_entropy_with_logits(logits = self.logits, 
            #                                             targets = self.labels,
            #                                             pos_weight = self.weights_logits * self.weights_labels))
            # self.cross_entropy_test4 = tf.reduce_sum(
            #     tf.nn.weighted_cross_entropy_with_logits(logits = self.logits, 
            #                                             targets = self.labels,
            #                                             pos_weight = self.weights_labels))


            # .reduce_sum( ( (self.weighted_logits*tf.log(self.weighted_predictions + 1e-9)) + ((1-self.weighted_logits) * tf.log(1 - self.weighted_predictions + 1e-9)) )  , name='xentropy' )    

             # + \
             #                     -self.weighted_labels_1 * tf.log(self.weighted_predictions_1)/4 + \
             #                     -self.weighted_labels_2 * tf.log(self.weighted_predictions_2)/4 + \
             #                     -self.weighted_labels_3 * tf.log(self.weighted_predictions_3)/4

            # self.cross_entropy = self.Xent(labels=self.labels, logits=self.logits, weights=self.class_weights)

            self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
            self._loss = tf.Variable(0.0, trainable=False, name='loss')
            self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
            self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')
            # self.loss = tf.reduce_mean(self.cross_entropy, name="loss")  
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)
            # self.summary_train_loss = tf.summary.scalar("train_loss", self.cost)
            # self.summary_valid_loss = tf.summary.scalar("valid_loss", self.cost)



        with tf.name_scope('metrics'):

            # self._precision, self._precision_update = tf.metrics.precision(labels=self.labels, predictions=tf.argmax(self.logits, 1), name='precision')
            # self._reset_precision_op = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='precision'))
            # self._precision = tf.Variable(0.0, trainable=False, name='precision')

            # self._recall, self._recall_update = tf.metrics.recall(labels=self.labels, predictions=tf.argmax(self.logits, 1), name='recall')
            # self._reset_recall_op = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='recall'))
            # self._recall = tf.Variable(0.0, trainable=False, name='recall')

            # self._f1_score = 2.0 * self._precision * self._recall / (self._precision + self._recall)

            # with tf.name_scope("accuracy"):
            #     self._accuracy, self._accuracy_update = tf.metrics.accuracy(labels=self.labels, predictions=self.predictions, name='accuracy')
            #     self._reset_accuracy_op = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy'))

            # self._auc_PR, self._auc_PR_update = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, curve='PR', name='auc_PR', )
            # self._reset_auc_PR_op = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='auc_PR'))

            # with tf.name_scope('count'):
            #     self._TP_0, self._TP_update_0  = tf.metrics.true_positives(labels=self.labels_0,  predictions=self.predictions_0, name="TP_0")
            #     # self._accuracy_0, self._accuracy_update_0 = tf.metrics.accuracy(labels=self.labels_0, predictions=self.predictions_0, name='accuracy_0', )
            #     self._reset_TP_0 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='TP_0'))


            # ACCURACY
            # with tf.name_scope('accuracy'):
            #     correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            #     self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            # with tf.name_scope("accuracy"):
            #     self._accuracy_0, self._accuracy_update_0 = tf.metrics.accuracy(labels=self.labels_0, predictions=self.predictions_0, name='accuracy_0', )
            #     self._reset_accuracy_op_0 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy_0'))

            #     self._accuracy_1, self._accuracy_update_1 = tf.metrics.accuracy(labels=self.labels_1, predictions=self.predictions_1, name='accuracy_1', )
            #     self._reset_accuracy_op_1 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy_1'))

            #     self._accuracy_2, self._accuracy_update_2 = tf.metrics.accuracy(labels=self.labels_2, predictions=self.predictions_2, name='accuracy_2', )
            #     self._reset_accuracy_op_2 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy_2'))

            #     self._accuracy_3, self._accuracy_update_3 = tf.metrics.accuracy(labels=self.labels_3, predictions=self.predictions_3, name='accuracy_3', )
            #     self._reset_accuracy_op_3 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy_3'))

            #     self._batch_accuracy = tf.reduce_mean([self._accuracy_0, self._accuracy_1, self._accuracy_2, self._accuracy_3], name='batch_accuracy')
            #     self._accuracy = tf.Variable(0.0, trainable=False, name='accuracy')
            #     self._accuracy_update = tf.assign(self._accuracy,  tf.reduce_mean([self._accuracy_update_0, self._accuracy_update_1, self._accuracy_update_2, self._accuracy_update_3]), name='accuracy_update' )
            #     self._reset_accuracy_op = tf.assign(self._accuracy, 0.0, name='reset_accuracy_op')


            # PRECISION
            # with tf.name_scope("precision"):
            #     # self._precision, self._precision_update = tf.metrics.precision(labels=tf.argmax(self.labels,1), predictions=tf.argmax(self.logits,1), name='precision', )
            #     # self._reset_precision_op = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='precision'))


            #     self._precision_0, self._precision_update_0 = tf.metrics.precision(labels=self.labels_0, predictions=self.predictions_0, name='precision_0', )
            #     self._reset_precision_op_0 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='precision_0'))

            #     self._precision_1, self._precision_update_1 = tf.metrics.precision(labels=self.labels_1, predictions=self.predictions_1, name='precision_1', )
            #     self._reset_precision_op_1 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='precision_1'))

            #     self._precision_2, self._precision_update_2 = tf.metrics.precision(labels=self.labels_2, predictions=self.predictions_2, name='precision_2', )
            #     self._reset_precision_op_2 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='precision_2'))

            #     self._precision_3, self._precision_update_3 = tf.metrics.precision(labels=self.labels_3, predictions=self.predictions_3, name='precision_3', )
            #     self._reset_precision_op_3 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='precision_3'))

            #     self._batch_precision = tf.reduce_mean([self._precision_0, self._precision_1, self._precision_2, self._precision_3], name='batch_precision')
            #     self._precision = tf.Variable(0.0, trainable=False, name='precision')
            #     self._precision_update = tf.assign(self._precision,  tf.reduce_mean([self._precision_update_0, self._precision_update_1, self._precision_update_2, self._precision_update_3]), name='precision_update' )
            #     self._reset_precision_op = tf.assign(self._precision, 0.0, name='reset_precision_op')




            # RECALL
            # with tf.name_scope("recall"):
            #     self._recall_0, self._recall_update_0 = tf.metrics.recall(labels=self.labels_0, predictions=self.predictions_0, name='recall_0', )
            #     self._reset_recall_op_0 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='recall_0'))

            #     self._recall_1, self._recall_update_1 = tf.metrics.recall(labels=self.labels_1, predictions=self.predictions_1, name='recall_1', )
            #     self._reset_recall_op_1 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='recall_1'))

            #     self._recall_2, self._recall_update_2 = tf.metrics.recall(labels=self.labels_2, predictions=self.predictions_2, name='recall_2', )
            #     self._reset_recall_op_2 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='recall_2'))

            #     self._recall_3, self._recall_update_3 = tf.metrics.recall(labels=self.labels_3, predictions=self.predictions_3, name='recall_3', )
            #     self._reset_recall_op_3 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='recall_3'))

            #     self._batch_recall = tf.reduce_mean([self._recall_0, self._recall_1, self._recall_2, self._recall_3], name='batch_recall')
            #     self._recall = tf.Variable(0.0, trainable=False, name='recall')
            #     self._recall_update = tf.assign(self._recall,  tf.reduce_mean([self._recall_update_0, self._recall_update_1, self._recall_update_2, self._recall_update_3]), name='recall_update' )
            #     self._reset_recall_op = tf.assign(self._recall, 0.0, name='reset_recall_op')



            with tf.name_scope("mAP"):
                # AUC PR = mAP
                self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )
                # self._reset_auc_PR_op_0 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='auc_PR_0'))

                self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )
                # self._reset_auc_PR_op_1 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='auc_PR_1'))

                self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )
                # self._reset_auc_PR_op_2 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='auc_PR_2'))

                self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )
                # self._reset_auc_PR_op_3 = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='auc_PR_3'))

                self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
                self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
                self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update' )
                # self._reset_auc_PR_op = tf.assign(self._auc_PR, 0.0, name='reset_auc_PR_op')

            # self.logits_0

            # self._auc_PR, self._auc_PR_update = tf.metrics.auc(labels=self.labels[:,0], predictions=self.predictions[:,0], curve='PR', name='auc_PR', )
            # self._reset_auc_PR_op = [self._reset_auc_PR_op_0, self._reset_auc_PR_op_1, self._reset_auc_PR_op_2, self._reset_auc_PR_op_3]


            # self._batch_auc_PR = tf.reduce_mean([self._reset_auc_PR_op_0, self._reset_auc_PR_op_1, self._reset_auc_PR_op_2, self._reset_auc_PR_op_3], name='batch_auc')
            # self._auc_PR = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')

            # self._auc_PR_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_auc_PR, name='confusion_matrix_update' )
            # self._reset_auc_PR = [self._reset_auc_PR_op_0, self._reset_auc_PR_op_1, self._reset_auc_PR_op_2, self._reset_auc_PR_op_3]




            # CONFUSION MATRIX
            self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
            self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
            self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update' )
            self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')











            # self._confusion_matrix, self._confusion_matrix_update = tf.contrib.metrics.confusion_matrix(labels=self.labels, predictions=self.predictions, num_classes=dataset.num_classes)
            # self._reset_confusion_matrix_op = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='confusion_matrix'))

        # with tf.name_scope('accuracy'):
        #     correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        #     self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.accuracy)
        #     self.summary_valid_accuracy = tf.summary.scalar("valid_accuracy", self.accuracy)
            # tf.summary.scalar("accuracy", self.accuracy)


        # self.confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes)


        # with tf.name_scope("summaries"):
        # self.train_merged = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        # self.valid_merged = tf.summary.merge([self.summary_valid_loss, self.summary_valid_accuracy])
        # self.valid_merged = tf.summary.merge_all()

        # self.merged = tf.summary.merge_all()
        
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')






    @property
    def loss(self):
        return self._loss


    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                # self._accuracy_update,
                # self._precision_update,
                # self._recall_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }
                

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                # self._reset_accuracy_op,
                # self._reset_precision_op,
                # self._reset_recall_op,
                # 'auc_PR': self._reset_auc_PR_op, # actually reset with local variable resetter
                # 'auc_PR_0': self._reset_auc_PR_op_0, # actually reset with local variable resetter
                # 'auc_PR_1': self._reset_auc_PR_op_1, # actually reset with local variable resetter
                # 'auc_PR_2': self._reset_auc_PR_op_2, # actually reset with local variable resetter
                # 'auc_PR_3': self._reset_auc_PR_op_3, # actually reset with local variable resetter
                'loss': self._reset_loss_op,
                }


    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,  
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,                  
                # 'mAP': self.network.mAP,
                'confusion_matrix': self._confusion_matrix,
                }





