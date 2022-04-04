# -*- coding: UTF-8 -*-
#没写oldpi网络的情况
from cv2 import merge
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow import keras
import time
import cv2
import os
import math
import sys


A_LR = 0.00005
A_DIM =11
UPDATE_STEPS = 10

class PPO:
    def __init__(self,loadpath):

#########################################Network#################################
        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512
        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4
        # Parameters for CNN
        self.img_size = 80  # input image size
        self.first_conv = [8, 8, self.Num_stackFrame, 32]
        self.second_conv = [4, 4, 32, 64]
        self.third_conv = [3, 3, 64, 64]
        self.first_dense = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], 11]

#########################################Train#################################
        self.ent_coef=0.01
        self.vf_coef=0.5


        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        self.load_path = loadpath

        self.epsilon=0.2

        self.Num_action = 11


#######################################Initialize Network###############################
        self.network()
        self.sess,self.saver,self.writer=self.init_sess()
        pass


    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):  # 初始化偏置项
        return tf.Variable(self.xavier_initializer(shape))

    # Xavier Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(2.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    def conv2d(self, x, w, stride):  # 定义一个函数，用于构建卷积层
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    # Xavier Weights initializer
    def normc_initializer(self,std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)
        return _initializer


    def network(self):
        tf.reset_default_graph()
        # Input------image
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.Num_stackFrame],name="image")
        self.x_normalize = (self.x_image - (255.0/2)) / (255.0/2)  # 归一化处理

        # Input------sensor
        self.x_sensor = tf.placeholder(tf.float32, shape=[None, self.Num_stackFrame, self.Num_dataSize],name="state")
        self.x_unstack = tf.unstack(self.x_sensor, axis=1)

        with tf.variable_scope('Network'):
            # Convolution variables
            with tf.variable_scope('CNN'):
                w_conv1 = self.weight_variable(self.first_conv)  # w_conv1 = ([8,8,4,32])
                b_conv1 = self.bias_variable([self.first_conv[3]])  # b_conv1 = ([32])

                # second_conv=[4,4,32,64]
                w_conv2 = self.weight_variable(self.second_conv)  # w_conv2 = ([4,4,32,64])
                b_conv2 = self.bias_variable([self.second_conv[3]])  # b_conv2 = ([64])

                # third_conv=[3,3,64,64]
                w_conv3 = self.weight_variable(self.third_conv)  # w_conv3 = ([3,3,64,64])
                b_conv3 = self.bias_variable([self.third_conv[3]])  # b_conv3 = ([64])

                h_conv1 = tf.nn.relu(self.conv2d(self.x_normalize, w_conv1, 4) + b_conv1)
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
                h_pool3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64])  # 将tensor打平到vector中

            with tf.variable_scope('LSTM'):
                # LSTM cell
                #TODO:看看LSTM的结构
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.Num_cellState)
                rnn_out, rnn_state = tf.nn.static_rnn(inputs=self.x_unstack, cell=cell, dtype=tf.float32)      
                rnn_out = rnn_out[-1]

            with tf.variable_scope('AC'):
                h_concat = tf.concat([h_pool3_flat, rnn_out], axis=1)
                #TODO:initializer
                x = tf.nn.relu(tf.layers.dense(h_concat, 512, name='lin',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))              
                logits =tf.nn.softmax( tf.layers.dense(x, self.Num_action, name='logits', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) )  )
                self.dist=tf.distributions.Categorical(logits)
                self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))[:,0]
                self.ac = self.dist.sample() 

        with tf.variable_scope('Train'):
            self.A = A = tf.placeholder(tf.float32,[None],'action')
            self.ADV = ADV = tf.placeholder(tf.float32, [None],"ADV")
            self.R = R = tf.placeholder(tf.float32, [None],"r")
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
            self.LR = LR = tf.placeholder(tf.float32, [])
            # Cliprange
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

            self.neglogpac= neglogpac= self.dist.log_prob(A)
            entropy = tf.reduce_mean(self.dist.entropy())

            vpred = self.vpred

            #Value___Clip
            vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE) 
            # Unclipped value
            vf_losses1 = tf.square(vpred - R)
            # Clipped value
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            # Calculate ratio (pi current policy / pi old policy)
            ratio = tf.exp(neglogpac-OLDNEGLOGPAC)                            #应该是为了去掉分母为零的风险
            # Defining Loss = - J is equivalent to max J
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

            # Total loss
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            grads_and_var = self.trainer.compute_gradients(loss)
            self._train_op = self.trainer.apply_gradients(grads_and_var)


        
        with tf.variable_scope('Record'):
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # # 运行时记录运行信息的proto
            # run_metadata = tf.RunMetadata()
            # 将配置信息和运行记录信息的proto传入运行过程，从而进行记录
            closs=tf.summary.scalar("Critic_loss", vf_loss)
            aloss=tf.summary.scalar("Actor_loss",pg_loss)
            merged=tf.summary.merge_all()  
            self.stats_list = [merged,pg_loss, vf_loss, entropy, approxkl, clipfrac,neglogpac,logits]

        pass



    def init_sess(self):
        # Initialize variables
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存  
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # Load the file if the saved file exists
        check_save = input('Load Model? (1=yes/2=no): ')
        if check_save == 1:
            # Restore variables from disk.
            saver.restore(sess, self.load_path + "/model.ckpt")
            print("Model restored.")
            #TODO:这里只是PPO.Num_training
            check_train = input('Inference or Training? (1=Inference / 2=Training): ')
            if check_train == 1:
                self.Num_start_training = 0
                self.Num_training = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/PPO/' + current_time 
        writer=tf.summary.FileWriter(log_dir, tf.get_default_graph())
        return sess,saver,writer


    def choose_action(self,observation_stack,state_stack):
        # s = s[np.newaxis,:]
        a = self.sess.run(self.ac,{self.x_image:observation_stack,self.x_sensor:state_stack})[0]
        #TODO:看看这个action是什么形式的
        return(a)

    def train(self,observation_stack,state_stack,returns,actions,train_step, lr, cliprange, values,rewards):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # neglogpacs=self.sess.run(self.dist.log_prob(actions),{self.x_image:observation_stack,self.x_sensor:state_stack})
        neglogpacs=self.sess.run(self.neglogpac,{self.x_image:observation_stack,self.x_sensor:state_stack,self.A:actions})
        
        td_map = {
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.x_image:observation_stack,
            self.x_sensor:state_stack
        }
        for start in range(UPDATE_STEPS):
            merged,pg_loss, vf_loss, entropy, approxkl, clipfrac,newneglogpacs ,logits,_=self.sess.run(self.stats_list + [self._train_op],td_map)

        with open('Data/PPO_Discrete.txt','a') as f:             
            f.write('oldpi_prob')
            f.write('\r\n')
            np.savetxt(f, neglogpacs, delimiter=',', fmt = '%s')
            f.write('pi_prob')
            f.write('\r\n')
            np.savetxt(f, newneglogpacs, delimiter=',', fmt = '%s')
            f.write('adv')
            f.write('\r\n')
            np.savetxt(f, advs, delimiter=',', fmt = '%s')
            f.write('actions')
            f.write('\r\n')
            np.savetxt(f, actions, delimiter=',', fmt = '%s')
            f.write('rewards')
            f.write('\r\n')
            np.savetxt(f, rewards, delimiter=',', fmt = '%s')
            f.write('logits')
            f.write('\r\n')
            np.savetxt(f, logits, delimiter=',', fmt = '%s')
            f.write('\r\n')

        self.writer.add_summary(merged, train_step) 

        pass

    def get_v(self,observation_stack,state_stack):
        # if observation_stack.ndim < 2:s = s[np.newaxis,:]
        return self.sess.run(self.vpred,{self.x_image:observation_stack,self.x_sensor:state_stack})[0]

    def save_model(self):
        # ------------------------------
        save_path = self.saver.save(
            self.sess, 'saved_networks/' + 'PPO_Discrete' + '_' + self.date_time + "/model.ckpt")
        # ------------------------------
        pass