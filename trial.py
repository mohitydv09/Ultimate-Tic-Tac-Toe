import torch
import numpy as np

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])


# c= np.array([1,2,3])
# d= np.array([4,5,6])

collec = (a,b)
print(type(collec))
c = np.array(collec)
d = torch.tensor(c)

print(c, d)