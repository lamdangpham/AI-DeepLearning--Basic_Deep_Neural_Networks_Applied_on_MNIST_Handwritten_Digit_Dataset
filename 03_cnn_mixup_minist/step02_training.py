import tensorflow as tf
import numpy as np
import os
import math
import scipy.io
import re
import time
import datetime
import sys
from sklearn import datasets, svm, metrics
from numpy import random
sys.path.append('./network/')
from model_conf import *


random.seed(1)
#=========================================== 01/ PARAMETERS
print("\n ==================================================================== SETUP PARAMETERS...")

tf.flags.DEFINE_string("OUT_DIR",    "./data/",     "Point to output directory")

tf.flags.DEFINE_integer("BATCH_SIZE",        100,    "Batch Size ")
tf.flags.DEFINE_integer("NUM_BATCH",         9000,    "Number of training epochs (default: 100)")
tf.flags.DEFINE_float  ("LEARNING_RATE",     1e-4,  "Learning rate")

tf.flags.DEFINE_boolean("allow_soft_placement", True,  "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
mixup_num = int(FLAGS.BATCH_SIZE/2)
#=========================================== 02/ Import Input Data
print("\n ==================================================================== IMPORT DATA...")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#=========================================  03/ TRAINING & SAVE
print("\n ==================================================================== TRAINING DATA...")
tf.reset_default_graph()
with tf.Graph().as_default():
    #option to setup memory 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=False)
    
    #option to setup session
    tf.set_random_seed(1)
    session_conf = tf.ConfigProto( allow_soft_placement=FLAGS.allow_soft_placement, 
                                   log_device_placement=FLAGS.log_device_placement
                                   #gpu_options = gpu_options
                                 )
    #Call session
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        print("\n =============== 01/ Instance Model")
        model = model_conf()  

        print("\n =============== 02/ Setting Training Options")
        print("\n + Adam optimizer ")
        print("\n + Learning Rate: {}".format(FLAGS.LEARNING_RATE))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step    = tf.Variable(0, name="global_step", trainable=False)
            optimizer      = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op       = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        print("\n =============== 03/ Setting Report ...")
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram ("{}/grad/hist".format(v.name), g)
                sparsity_summary  = tf.summary.scalar    ("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)
        loss_summary          = tf.summary.scalar("loss", model.loss)
        acc_summary           = tf.summary.scalar("accuracy", model.accuracy)
        train_summary_op      = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])


        print("\n =============== 04/ Setting Directory for Saving...")
        stored_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.OUT_DIR))
        print("+ Writing to {}\n".format(stored_dir))

        train_summary_dir = os.path.join(stored_dir, "summaries", "train")   
        print("+ Training summary Writing to {}\n".format(train_summary_dir))

        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        checkpoint_dir = os.path.abspath(os.path.join(stored_dir, "checkpoints"))
        print("+ Checkpoint Dir: {}\n".format(checkpoint_dir))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        best_model_dir = os.path.join(stored_dir, "model")
        print("+ Best model Dir: {}\n".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        print("\n =============== 05/ Creating Saver...")
        saver = tf.train.Saver(tf.global_variables())

        best_model_files     = os.path.join(best_model_dir, "best_model")
        best_model_meta_file = os.path.join(best_model_dir, "best_model.meta")
        print("+ Best Model Files: {}\n".format(best_model_files))
        print("+ Best Model Meta File: {}\n".format(best_model_meta_file))

        if os.path.isfile(best_model_meta_file):
            print("\n + Latest Model Loaded from dir: {}" .format(best_model_dir))
            saver = tf.train.import_meta_graph(best_model_meta_file)
            saver.restore(sess, best_model_files)
        else:
            print("\n + New Model Initialized")
            sess.run(tf.global_variables_initializer())

        #========================= Training process 
        is_training = True
        is_testing  = True

        if(is_training):
            print("\n ============= TRAINING ....")
            for i in range(int(FLAGS.NUM_BATCH)):
                # Get mnist data
                batch = mnist.train.next_batch(FLAGS.BATCH_SIZE) 
                seq_x = batch[0] #100:784
                seq_x = np.reshape(seq_x, [int(FLAGS.BATCH_SIZE), 28, 28, 1])
                seq_y = batch[1] #100:10

                #Mixup data
                X1 = seq_x[:mixup_num]
                X2 = seq_x[mixup_num:]
                y1 = seq_y[:mixup_num]
                y2 = seq_y[mixup_num:]

                # Betal dis
                b   = np.random.beta(0.4, 0.4, mixup_num)
                X_b = b.reshape(mixup_num, 1, 1, 1)
                y_b = b.reshape(mixup_num, 1)

                xb_mix   = X1*X_b     + X2*(1-X_b)
                xb_mix_2 = X1*(1-X_b) + X2*X_b
                yb_mix   = y1*y_b     + y2*(1-y_b)
                yb_mix_2 = y1*(1-y_b) + y2*y_b

                # Uniform dis
                l   = np.random.random(mixup_num)
                X_l = l.reshape(mixup_num, 1, 1, 1)
                y_l = l.reshape(mixup_num, 1)

                xl_mix   = X1*X_l     + X2*(1-X_l)
                xl_mix_2 = X1*(1-X_l) + X2*X_l
                yl_mix   = y1* y_l    + y2 * (1-y_l)
                yl_mix_2 = y1*(1-y_l) + y2*y_l

                #Could use augmentation data from betal/uniform distribution
                seq_x = np.concatenate((xl_mix, xl_mix_2), 0)
                seq_y = np.concatenate((yl_mix, yl_mix_2), 0)


                feed_dict= {model.input_layer_val:   seq_x,
                            model.expected_classes:  seq_y,
                            model.mode: True
                           }
                [ _, step, summaries, loss, accuracy] = sess.run([train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))   

                train_summary_writer.add_summary(summaries, step)

            best_model_files = os.path.join(best_model_dir, "best_model")
            saved_path       = saver.save(sess, best_model_files)
            print("\n Save model after training ")
         
        
        #========================= Testing process 
        if(is_testing):
            print("\n ============= TESTING ....")
            seq_x_test = mnist.test.images
            [nSamle, nDim] = np.shape(seq_x_test)
            seq_x_test = np.reshape(seq_x_test, [nSamle, 28, 28, 1])
            seq_y_test = mnist.test.labels
            feed_dict= {model.input_layer_val:   seq_x_test,
                        model.expected_classes:  seq_y_test,
                        model.mode: False
                       }
            [accuracy] = sess.run([model.accuracy], feed_dict)
            print("Acc {:g}".format(accuracy))   
