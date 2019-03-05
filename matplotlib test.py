import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3, 3, 50)
y1 = x*2-1
y2 = x**3
plt.figure(num = 2, figsize= (8,5))#color types:
plt.plot(x,y1,color='red',linewidth=1.2,linestyle = '-.',label = 'up')#linestyle: '-','--','-.',':'
plt.plot(x,y2,label ='down')
plt.xlim(-1,2),plt.ylim(-2,3)
plt.xlabel('wo shi shui')
'''new_tick = np.linspace(-1,2,5)  #-1 to 2, total 5 numbers.
plt.xticks(new_tick)'''
plt.yticks([-2, -1.8, -1, 1.22, 3],['very',' bad', 'normol','good', 'very good' ])#change the tick
ax = plt.gca() # gca:get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')# change the position of ticks
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.legend(loc='best')
plt.show()
