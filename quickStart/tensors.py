import torch

x = torch.empty(3)
x = torch.empty(3,2)
x = torch.empty(3,4)

x=torch.rand(1,2)

x = torch.ones(2,2,dtype=torch.float16)

x = torch.tensor([2.5, 0.1])
x = torch.tensor([[2.5, 0.1],[2,3]])

x = torch.rand(2,2)
y = torch.rand(2,2)

#OPERATIONS
# z =x+y
#z = torch.add(x,y)
#y.add_(x) #_ işareti modifiye ediyor y'yi

#z = x - y
#z = torch.sub(x,y)

#z = x*y
#z = torch.mul(x,y)

#z = x/y
#z = torch.div(x,y)

x = torch.rand(5,3)
#print(x)
#print(x[1,1])
#print(x[1,1].item()) # 1D tensor olduğunda kullanılabilir tensor değerini döndürüyor
#print(x[1,...])
#print(x[1,:])

x = torch.rand(2,2)
#print(x)
#y = x.view(4)
#y = x.view(2,2)
#y = x.view(-1,4)
#print(y.shape)
#print(y.size())
#print(y.size(0))


"""
###################
"""

import numpy as np

a = torch.ones(5)
"""
print(a)
b = a.numpy()
print(b)  ### Bu şekilde GPU CPU'da aynı hafızayı paylaşıyorlar verileri ezmemeye dikkat etmek gerekiyor

a.add_(4)
print(a)
print(b)
"""

a = np.ones(5)
"""
print(a)
b = torch.from_numpy(a)
print(b)

a = a+5
#b.add_(4)
print(a)
print(b)
"""

"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    print(z)
    z = z.to("cpu")
    print(z)
"""

x = torch.ones(5,requires_grad=True)
print(x)   ### x değerinin gradyenine ihtiyaç olacak bellekte gerekli bilgileri depola