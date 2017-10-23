from numpy import array, zeros, ones, argmax, log

def loadData():
	data = ["my dog has flea help please", "maybe not take him to dog park stupid", "my dalmation is so cute I love him",
			"stop posting stupid worthless garbage", "mr licks ate my steak how to stop him", "quit buying worthless dog food stupid"]
	data = [item.strip().split(" ") for item in data]
	classVec = [0, 1, 0, 1, 0, 1]
	return data, classVec

def createVocabList(data):
	#@data: list of string
	vocabSet = set()
	for i in range(len(data)):
		vocabSet = set(data[i]) | vocabSet
	return list(vocabSet)

def words2Vec(data, vocabList):
	#word set model
	#@data: list of string
	dataCount = len(data)
	featureCount = len(vocabList)
	dataMatrix = zeros((featureCount, dataCount))
	for i in range(dataCount):
		for word in data[i]:
			if word not in vocabList: 
				print "word ", word, "not in vocab list"
			else:
				currentIndex = vocabList.index(word)
				dataMatrix[currentIndex, i] = 1
	return dataMatrix

def words2Bag(data, vocabList):
	#word bag model
	#@data: list of string
	dataCount = len(data)
	featureCount = len(vocabList)
	dataMatrix = zeros((featureCount, dataCount))
	for i in range(dataCount):
		for word in data[i]:
			if word not in vocabList: 
				print "word ", word, "not in vocab list"
			else:
				currentIndex = vocabList.index(word)
				dataMatrix[currentIndex, i] += 1
	return dataMatrix

def trainNB0(trainMatrix, labelList):
	#value has only 1 and 0
	featureNumber = len(trainMatrix)
	dataNumber = len(trainMatrix[0])
	#First P(Ci) noted as pPrior
	labelCount = {}
	pPrior = {}
	for label in labelList:
		labelCount[label] = labelCount.get(label, 0) + 1
	for label in labelCount.keys():
		pPrior[label] = labelCount[label]*1.0/len(labelList)
	#Second P(wi|ci) noted as pLikelihood
	pNum = {}
	pDenom = {}
	logpLikelihood = {}
	for label in labelCount.keys():
		pNum[label] = ones((featureNumber,1))
		pDenom[label] = 2.0
	for i in range(dataNumber):
		currentData = trainMatrix[..., i:i+1]
		pNum[labelList[i]] += currentData
		pDenom[labelList[i]] += currentData[..., 0].sum()
	for label in labelCount.keys():
		logpLikelihood[label] = log(pNum[label]*1.0) - log(pDenom[label])
	return logpLikelihood, pPrior

def classifyNB(testMatrix, logpLikelihood, pPrior):
	#require: testMatrix should have same shape as logpLikelihood.
	numOfData = len(testMatrix[0])
	featureNumber = len(testMatrix)
	labelList = pPrior.keys()
	labelNumber = len(labelList)
	pPost = zeros((labelNumber, numOfData))
	bestLabel = []
	for i in range(labelNumber):
		label = labelList[i]
		pPost[i] = (testMatrix * logpLikelihood[label]).sum(axis=0)+log(pPrior[label])
	for i in range(numOfData):
		bestLabel.append(labelList[argmax(pPost[..., i])])
	return bestLabel

if __name__ == '__main__':

	data, classVec = loadData()
	vocabList = createVocabList(data)
	dataMatrix = words2Bag(data, vocabList)
	pLikelihood, pLabel = trainNB0(dataMatrix, classVec)

	testData = ["you are a stupid garbage", "I love my dalmation"]
	testData = [item.strip().split(" ") for item in testData]
	dataMatrix = words2Bag(testData, vocabList)
	print classifyNB(dataMatrix, pLikelihood, pLabel)
