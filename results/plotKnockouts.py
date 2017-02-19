#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from glob import glob
import pickle
import numpy as np

totalLayers = 5
totalPercentages = 10
totalTrials = 1
metric = "top5performance"

dataArray = [[[] for  _ in range(totalPercentages)] for  _ in range(totalLayers)]

# print (dataArray)

folderName = sys.argv[1]
for layerNum in range(1,1+totalLayers):
	for percent in range(0,0+totalPercentages):
		for trialNum in range(1,1+totalTrials):
			tmp = pickle.load(open(folderName+"/alexnet-conv_"+str(layerNum)+"-nodeKnockout0."+str(percent)+"0-num"+str(trialNum)+"-num1.p","rb")); 
			# print("alexnet-conv_"+str(layerNum)+"-nodeKnockout0."+str(percent)+"0-num1-num1.p:  ",end='')
			# print(tmp[metric])
			dataArray[layerNum-1][percent].append(tmp[metric])
	# print("")
# print(dataArray)

fig, ax = plt.subplots()
for layerNum in range(totalLayers):
	ax.plot(np.arange(totalPercentages)/10.0,dataArray[layerNum], label='conv_'+str(layerNum+1))

ax.legend(loc=0)
ax.set_xlim([-0.01,1.01])
ax.set_ylim([-0.01,0.81])
ax.set_xlabel("Proportion of Nodes Knocked Out")
ax.set_ylabel("Top-5 Performance")
plt.tight_layout()
plt.savefig("knockout.pdf")
