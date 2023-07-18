import numpy as np 

snapnum=8
f=np.genfromtxt("obdata"+str(snapnum).zfill(3)+".dat",names=True,comments='#')
print f['m']
print f['phi']
print f['uperr']
print f['lowerr']
