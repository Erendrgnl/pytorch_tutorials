import torch

x = torch.randn(3,requires_grad=True)
#print(x)

y = x+2
#print(y)

z = y*y*2
"""
#print(z)
z = z.mean() ##scalar output olmazsa hata olur ve backward içine veri gödernemiz gerekir
#print(z)

z.backward() #dZ/dX
print(x.grad)
"""
"""
v = torch.tensor([1,2,1],dtype=torch.float32)
z.backward(v)
print(x.grad)
"""

#x.requires_grad(False)
#x.detach()
#with torch.no_grad():

#print(x)
#y = x.detach()
#print(y) ## no grad req

#with torch.no_grad():
#    y = x+2
#    print(y)

"""
weights = torch.ones(4,requires_grad=True)
for epoch in range(5): #for epoch in range(5):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() ## her iterasyonda gradyenleri sıfırlamak gerekiyor yoksa birikiyor
"""

weights = torch.ones(4,requires_grad=True)
optimizer = torch.optim.SGD(weights,lr=0.01)
optimizer.step()
optimizer.zero_grad() ## Her iterasyonda sıfırlamak gerekiyor