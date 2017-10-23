from math import log
from numpy import delete, array
from plotTree import createPlot, getExampleTree

def calculateEntropy(data):
	labels = data[-1]
	labelcount = {}
	entropy = 0.0
	labellen = len(labels)
	for label in labels:
		labelcount[label] = labelcount.get(label, 0) + 1
	labelcount[label] = labelcount[label]*1.0/labellen
	for label in labelcount.keys():
		entropy += -labelcount[label]*log(labelcount[label], 2)
	return entropy

def splitDataset(data, featureIndex, featureValue):
	dataCount = len(data[0])
	deleted = []
	for i in range(dataCount):
		if data[featureIndex, i] != featureValue:
			deleted.append(i)
	remainData = delete(data, deleted, axis=1)
	remainData = delete(remainData, featureIndex, axis=0)
	return remainData

def chooseBestFeatureToSplit(data):
	print data
	dataCount = len(data[0])
	featureCount = len(data)-1
	baseEntrophy = calculateEntropy(data)
	bestInfoGain = 0.0
	bestFeature = -1
	for featureIndex in range(featureCount):
		allValue = set(data[featureIndex, ...])
		newEntrophy = 0.0
		for featureValue in allValue:
			remainData = splitDataset(data, featureIndex, featureValue)
			remainEntrophy = calculateEntropy(remainData)
			prob = len(remainData[0])*1.0/dataCount
			newEntrophy += -prob*remainEntrophy
		infoGain = baseEntrophy - newEntrophy
		if infoGain > bestInfoGain:
			bestFeature = featureIndex
			bestInfoGain = infoGain
	print "Index", featureIndex
	return featureIndex

def majorityCount(data):
	#require: data with more than one label but no feature
	totalData = len(data[0])
	labelCount = {}
	maxCount = 0
	maxLabel = None
	for label in data[0]:
		labelCount[label] = labelCount.get(label, 0) + 1
		if labelCount[label] > maxCount:
			maxCount = labelCount[label]
			maxLabel = label
		if maxCount > totalData*1.0/2:
			return maxLabel
	return maxLabel

def isLabelAllSame(labels):
	firstLabel = labels[0]
	for label in labels:
		if label != firstLabel:
			return False
	return True

# def createBranch(data):
# 	if isLabelAllSame(data):
# 		return data[-1, 0] # a label to return
# 	else:
# 		currentFeatureIndex = chooseBestFeatureToSplit(data)
# 		allValue = set(data[currentFeatureIndex])
# 		for value in allValue:
# 			remainData = splitDataset(data, currentFeatureIndex, value)


def createTree(data, featureNames):
	labels = data[-1]
	if isLabelAllSame(labels): 
		return labels[0] # return a label
	if len(data) <= 1:
		return majorityCount(data) # return a label
	currentFeatureIndex = chooseBestFeatureToSplit(data)
	currentFeatureName = featureNames[currentFeatureIndex]
	myTree = {currentFeatureName: {} }
	del(featureNames[currentFeatureIndex])

	allValue = set(data[currentFeatureIndex])
	for value in allValue:
		branchFeatureNames = featureNames[:]
		remainData = splitDataset(data, currentFeatureIndex, value)
		myTree[currentFeatureName][value] = createTree(remainData, branchFeatureNames)
	return myTree 

def kTrees_classifier(vector, myTree, featureNames):
	#requirement: tree should at least use one feature to devide.
	currentFeature = myTree.keys()[0]
	currentFeatureIndex = featureNames.index(currentFeature)
	secondDict = myTree[currentFeature]
	currentValue = vector[currentFeatureIndex, 0]
	for key in secondDict.keys():
		if key == currentValue:
			if type(secondDict[key])!=type(dict()):
				return secondDict[key]
			else:
				return kTrees_classifier(vector, secondDict[key], featureNames)

#Use pickle(in Python) to store ktree in RAM


# myTree = getExampleTree()[0]
# features = ['no surface', 'flippers', 'head']
# data = array([[1], [0], [1]])
# print kTrees_classifier(data, myTree, features)

a=array([['young','young','pre','pre','presbyopic','presbyopic']
,['myope','hyper','myope','hyper','myope','hyper']
,['hard','hard','hard','no,lenses','hard','no,lenses']])

b=array([['young','young','pre','pre','presbyopic','presbyopic']
,['myope','hyper','myope','hyper','myope','hyper']
,['soft','soft','soft','soft','no,lenses','soft']])

chooseBestFeatureToSplit(b)
