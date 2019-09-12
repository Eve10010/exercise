#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:44:44 2019

@author: eve
"""
import numpy as np
import unicodedata  # 处理ASCii码的包
#import pickle
import os
import pickle
import math
lines=[]
os.system('sudo ovs-ofctl dump-flows -O Openflow13 s1 >& temp_data.txt')
file=open("temp_data.txt","r")
for i in file:
    lines.append(str(i))
file.close()

#parameters' name
name_list = ['port','cookie','duration','table','n_packets','n_bytes','idle_timeout',
           'priority','in_port','dl_src','dl_dst','nw_src','nw_dst','tp_src',
           'tp_dst','tcp_flags','actions']

#delete 'OFPST_FLOW reply (OF1.3) (xid=0x2):'
new_delete=[]
temp=[]
length_lines=len(lines)
for j in range(length_lines):   
    if lines[j]== 'OFPST_FLOW reply (OF1.3) (xid=0x2):\n':
        j=j+1
    else:       
        temp=lines[j]
        temp=temp.replace(' action',',action')
        temp=temp.replace(' ','')
        temp=temp.replace('\n','')
        temp=temp.split(',')
        new_delete.append(temp)

length_new_delete=len(new_delete)

final_list=[]
hash0=[]
hash1=[]

for m in range(length_new_delete):
    length_raw=len(new_delete[m])
    dic={}
    for n in range(length_raw):
        if '=' in new_delete[m][n]:
            sp=new_delete[m][n].split('=')
            if sp[0] == 'duration':
                sp[1]=sp[1].replace('s','')   #delete's'
            if sp[0] == 'cookie':#hex string turns into octimal*(cookie)
                sp[1]=int(sp[1],16)
                
            for l in range(9,11):
                if sp[0] == name_list[l]:
                    sp[1]=sp[1].split(':')
                    for h in range(len(sp[1])):
                        sp[1][h]=int(sp[1][h],16)#hex string turns into octimal(dl_src,dl_dst)  
                    length_sp1 = len(sp[1])
                    result = 0
                    for i in range(length_sp1):
                        result += sp[1][length_sp1-i-1]*(256**i)
                    sp[1]=result
                    break
    
            for k in range(11,13):  
                if sp[0] == name_list[k]:
                    sp[1]=sp[1].split('.')
                    for p in range(len(sp[1])):
                        sp[1][p]=int(sp[1][p],16)#hex string turns into octimal(_src,dst)
                    length_sp1 = len(sp[1])
                    result = 0
                    for i in range(length_sp1):
                        result += sp[1][length_sp1-i-1]*(256**i)
                    sp[1]=result
                    break
            if sp[0] == 'actions':
                sp[1] = sp[1].split(':')
                hash0=hash(sp[1][0])
                hash1=hash(float(sp[1][1]))
                sp[1]=hash0
            
            dic['port']=hash1    
            dic[sp[0]]=sp[1]
            
    final_list.append(dic)


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


def getnum(string):
    if is_number(string):
        return float(string)
    

#extract into a matrix(ignore 'action' for a while)   
mymat=np.zeros(shape=(17,length_new_delete))   

for a in range(17):
    for b in range(length_new_delete):
        if name_list[a] in final_list[b].keys():
            mymat[a][b]=getnum(final_list[b][name_list[a]])
            if abs(mymat[a][b])>1000:
                mymat[a][b]=math.log(10,abs(mymat[a][b]))
        else:
            pass


'''for c in range(17):
    for d in range(length_new_delete):
        mymat[c][d]=(mymat[c][d]-min(mymat[:,d]))/(max(mymat[:,d])-min(mymat[:,d]))'''
        
fs = open('mymat.pkl','wb')

pickle.dump(mymat,fs)       

fs.close()

   





