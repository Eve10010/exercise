#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:00:38 2019

@author: eve
"""
import os
import pickle
from Net import DeepQNet
def run():
    print("enter run")
    step=0
    for episode in range(300):
        print("enter episode:",episode)
        os.system('./datahandle.py')
        df=open('mymat.pkl','rb')
        observation=pickle.load(df)[:,0]
        df.close() 
        '''>& temp_observation.txt
        file = open("temp_observation.txt","r")
        for i in file:
            for j in i:
            observation.append(float(j))
        file.close()'''
        #for i in range(observation[0]):
        print(" ready to run while True")
        while True:
            action = RL.choose_action(observation)
            print('######',action,'########')
            observation_,reward,done = RL.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 50) and (step % 5 == 0):
                RL.learn()
            observation= observation_
            if done:
                break
            step +=1
    return action
if __name__ =="__main__":
    print("main is starting")
    RL = DeepQNet(n_actions=2,n_features=17,learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iteration=50,  # 每 50 步替换一次 target_net 的参数
                      memory_size=200,# 记忆上限
                      )   
    action = run()
    if action == 1:
        os.system('curl -X POST -d @ddos.json http://localhost:8080/wm/staticflowentrypusher/json')
    RL.plot_cost()      