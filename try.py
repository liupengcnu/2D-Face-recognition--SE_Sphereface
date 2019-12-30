import numpy as np

landmark = {}
with open('data/lfw_landmark.txt') as f:
    lambmark_lines = f.readlines()
#print(lambmark_lines[0])
for line in lambmark_lines[0]:
    l = line.replace('\n', '').split('\t')
    #print (l)
    #landmark[l[0]] = [int(k) for k in l[1:]]
    #print (landmark[l[0]])

a = np.arange(-1, 1, 0.005)
print (a)
print (a.shape)
