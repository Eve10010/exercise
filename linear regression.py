import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
y = x**3+x**2+0.2*torch.rand(x.size()) #noisy addition

#data show
'''plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()'''

#use torch to build net
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__() #fooparent success
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net = Net(n_feature=1,n_hidden=10,n_output=1)
'''print(net)'''   #show net

#start training

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_func=torch.nn.MSELoss()

#paint
plt.ion() #开始交互模式
plt.show()
for t in range(100):
    prediction = net(x)#output prediction 
    loss = loss_func(prediction,y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        
plt.ioff()

plt.show()
    
    
    

        
    





        
        
    



