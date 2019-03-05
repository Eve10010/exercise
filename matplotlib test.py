import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3, 3, 50)
y1 = x*2-1
'''y2 = x**3
plt.figure(num = 2, figsize= (8,5))#color types:
plt.plot(x,y1,color='red',linewidth=1.2,linestyle = '-.',label = 'up')#linestyle: '-','--','-.',':'
plt.plot(x,y2,label ='down')
plt.xlim(-1,2),plt.ylim(-2,3)
plt.xlabel('wo shi shui')
new_tick = np.linspace(-1,2,5)  #-1 to 2, total 5 numbers.
plt.xticks(new_tick)
plt.yticks([-2, -1.8, -1, 1.22, 3],['very',' bad', 'normol','good', 'very good' ])#change the tick
#axis handle
##################
ax = plt.gca() # gca:get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')# change the position of ticks
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.legend(loc='best')
plt.show()'''
#Annotation
####################
plt.figure(num = 1)
plt.plot(x,y1)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
#note dot
x0 = 1
y0 = 1
plt.scatter(x0,y0)
#dotted line
plt.plot([x0,x0,],[0,y0,],'k--')
plt.annotate('2*x0-1 = 1',xy = (x0,y0),xycoords = 'data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.text(-1,-1,'I\'m happy')
    # tick change
########################
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7, zorder=2))
 # 其中label.set_fontsize(12)重新调节字体大小，bbox设置目的内容的透明度相关参，
# facecolor调节 box 前景色，edgecolor 设置边框， 本处设置边框为无，alpha设置透明度.
plt.show ()
########
#scatter
n = 1024    # data size
X = np.random.normal(0, 1, n) # 每一个点的X值
Y = np.random.normal(0, 1, n) # 每一个点的Y值
T = np.arctan2(Y,X) # for color value
plt.figure(num=2)
plt.scatter(X,Y,c=T,alpha=0.5)
plt.yticks(())  # ignore yticks
###########
#bar figure
'''plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
for x, y in zip(X, Y1):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')'''

#############多合一

##method1
plt.subplot(2,2,1)#两行两列第一个图，以此类推
## method 2
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

plt.show()