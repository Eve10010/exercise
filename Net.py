#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:34:05 2019

@author: eve

"""
import tensorflow as tf
import psutil
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

class DeepQNet:
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.1,e_greedy=0.9,replace_target_iteration=50,
                 memory_size=200,batch_size=51,e_greedy_increment=None,out_gragh=False):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iteration = replace_target_iteration
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max#是否开启探索模式
        
        self.learn_step_counter = 0  #判断是否更换q_target参数
        self.memory = np.zeros((self.memory_size,n_features*2+2))#500 raws,observation+action+reward+observation_
        
        self.build_net()
        
        t_params = tf.get_collection('target_net_params')  # 提取 target_net 的参数
        e_params = tf.get_collection('eval_net_params')   # 提取  eval_net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # 更新 target_net 参数
        
        self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())
        
        self.cost_his=[]
        self.error_counter=0
    
        
        
    def build_net(self):
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name='q_target')
        
        with tf.variable_scope('eval_net'):
            c_names,n_l1,w_init,b_init = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES],10,\
            tf.random_normal_initializer(0,0.3),tf.constant_initializer(0.1)
                       
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_init,collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer=b_init,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
            
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_init,collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions],initializer=b_init,collections=c_names)
                self.q_eval= tf.matmul(l1,w2)+b2
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
        #---------------build q_target_net,s_ is input----------------------
        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_init,collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer=b_init,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
        
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_init,collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions],initializer=b_init,collections=c_names)
                self.q_next = tf.matmul(l1,w2)+b2
      
    def store_transition(self,s,a,r,s_): 
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
            
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter% self.memory_size
        self.memory[index,:] = transition
         
        self.memory_counter += 1
         
    def choose_action(self,observation):
          #observation shape:(1,n_feature)
        observation = observation[np.newaxis,:]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})#运行时i的数据来自于观测值字典
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action
    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])
          
    def step(self,action):
        cpuusage= psutil.cpu_percent(0)
        if action==0:
            os.system('./datahandle.py')
            df=open('mymat.pkl','rb')
            observation_=pickle.load(df)[:,0]
            df.close()
            s_=observation_
            if cpuusage<39:                        
                reward = 0
                done = True
                self.error_counter=self.error_counter+1   
            else:
                reward = -1
                done = False
        if action ==1:
            os.system('./datahandle.py')
            df=open('mymat.pkl','rb')
            observation_=pickle.load(df)[:,0]
            df.close()
            s_=observation_
            if cpuusage<39:
                reward = -1
                done = False
            else:
                reward = 0
                done = True
        return s_,reward,done
            
            
    def learn(self):
          #替换target_net参数
        if self.learn_step_counter % self.replace_target_iteration==0:
            self._replace_target_params()
              
              #抽取记忆
        if self.memory_counter>self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
          
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            })          
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
          
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost) 
    
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        