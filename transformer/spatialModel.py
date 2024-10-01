import torch

class MLP(torch.nn.Module):
	def __init__(self,nIn,nOut,Hidlayer, withReLU):
		super(MLP, self).__init__()
		numHidlayer=len(Hidlayer)
		net=[]
		net.append(torch.nn.Linear(nIn,Hidlayer[0]))
		if withReLU:
			net.append(torch.nn.ReLU())
		for i in range(0,numHidlayer-1):
			net.append(torch.nn.Linear(Hidlayer[i],Hidlayer[i+1]))
			if withReLU:
				net.append(torch.nn.ReLU())
		net.append(torch.nn.Linear(Hidlayer[-1],nOut))#
		self.mlp=torch.nn.Sequential(*net)
	def forward(self,x):
		return self.mlp(x)