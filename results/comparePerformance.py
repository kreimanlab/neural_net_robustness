#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from glob import glob
import pickle
import numpy as np
import pandas as pd

from scipy.stats import ranksums,mannwhitneyu

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rc('axes', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
matplotlib.rc('xtick', labelsize=14) 

colorList = sns.color_palette()+sns.color_palette("husl", 8)[:2]+sns.color_palette("Set2", 8)

###########################################################
# plot settings

metric = "top5performance"
# metric = "top1performance"
perturbationName = "mutateGaussian"
# perturbationName = "nodeKnockout"
# perturbationName = "synapseKnockout"
networkName = "alexnet"
# networkName = "vgg16"
# maxPercent = 1.0
# maxPercent = 9999
statOnly = True
# statOnly = False

##############################################################

dataFrames = []
nameList = []
for csvFile in sys.argv[1:]:
	dataFrames.append(pd.read_csv(csvFile))
	nameList.append(csvFile.split("_")[1])

layerNameList = sorted(list(set(dataFrames[0]["layer"].values)))
percentList = sorted(list(set(dataFrames[0][dataFrames[0]["layer"]==layerNameList[0]]["percent"].values)))
percentList = [float("%.02f" % i) for i in percentList]
print (layerNameList)
print (percentList)

for thisLayerName in layerNameList:
	print(thisLayerName,"\n============================================")
	for thisPercent in percentList:
		print ("perturb",thisPercent,":")
		tmpList = []
		for i in range(2):
			print (nameList[i],np.mean(dataFrames[i][dataFrames[i]["layer"]==thisLayerName][dataFrames[i]["percent"]==thisPercent]["performance"].values))
			tmpList.append(dataFrames[i][dataFrames[i]["layer"]==thisLayerName][dataFrames[i]["percent"]==thisPercent]["performance"].values)
