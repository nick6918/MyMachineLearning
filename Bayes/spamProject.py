from BernoulliBayes import createVocabList, words2Bag, words2Vec, trainNB0, classifyNB
from numpy import delete, zeros
import random

def parseText(TextAsString):
	from re import split
	listOfWords = split(r"\W*", TextAsString)
	return [token.lower() for token in listOfWords]

def getAllFiles(direction):
	from os import listdir, path
	listOfFiles = []
	listOfItems = listdir(direction)
	for item in listOfItems:
		currentPath = direction + "/" + item
		if path.isdir(currentPath):
			listOfFiles.extend(getAllFiles(currentPath))
		else:
			listOfFiles.append(currentPath)
	return listOfFiles

def parseFiles(allFiles):
	data = []
	labelList = []
	for file in allFiles:
		# fp = open(file, "r")
		# allTexts = fp.read()
		# fp.close()
		with open(file, "r") as fp:
			allTexts = fp.read()
		listOfWords = parseText(allTexts)
		labelList.append(int("spam" in file))
		data.append(listOfWords)
	return data, labelList

#createVocabList
#words2Bag / words2Vec

def chooseRandomData(testRatio, dataMatrix, labelList):
	numOfData = len(dataMatrix[0])
	featureNumber = len(dataMatrix)
	testDataNumber = int(numOfData*testRatio)
	testDataMatrix = zeros((featureNumber, testDataNumber))
	testDataIndexes = random.sample(range(numOfData), testDataNumber)
	testDataIndexes.sort()
	trainLabelList = []
	testLabelList = []
	for i in range(testDataNumber):
		testDataMatrix[..., i] = dataMatrix[..., testDataIndexes[i]]
	for i in range(numOfData):
		if i in testDataIndexes:
			testLabelList.append(labelList[i])
		else:
			trainLabelList.append(labelList[i])
	trainDataMatrix = delete(dataMatrix, testDataIndexes, axis=1)
	return trainDataMatrix, testDataMatrix, trainLabelList, testLabelList, testDataIndexes

def testInIteration(times, testRatio, dataMatrix, labelList, allFiles, AlarmLabels):
	totalWrongRate = 0.0
	totalFalsePositiveRate = 0.0
	totalFalseNegativeRate = 0.0
	for i in range(times):
		print "******************************"
		print "Start ", i, "th test:"
		trainDataMatrix, testDataMatrix, trainLabelList, testLabelList, testDataIndexes = chooseRandomData(testRatio, dataMatrix, labelList)
		
		logpLikelihood, pPrior = trainNB0(trainDataMatrix, trainLabelList)
		falseEventCount = 0
		trueEventCount = 0
		for label in testLabelList:
			if label in AlarmLabels:
				trueEventCount += 1
			else:
				falseEventCount += 1
		totalCount = falseEventCount + trueEventCount

		bestLabel = classifyNB(testDataMatrix, logpLikelihood, pPrior)
		falsePositiveCount = 0
		falseNegativeCount = 0
		for j in range(totalCount):
			if bestLabel[j] != testLabelList[j]:
				if bestLabel[j] in AlarmLabels:
					falsePositiveCount += 1
				else: 
					falseNegativeCount += 1
				print allFiles[testDataIndexes[j]], " has wrong label: ", bestLabel[j], " correct: ", testLabelList[j]
		falsePositiveRate = falsePositiveCount*1.0/totalCount
		falseNegativeRate = falseNegativeCount*1.0/totalCount
		wrongRate = falsePositiveRate+falseNegativeRate
		print i, "th test, wrong number: ", (falsePositiveCount+falseNegativeCount), "with rate: ", wrongRate
		totalWrongRate += wrongRate
		totalFalsePositiveRate += falsePositiveRate
		totalFalseNegativeRate += falseNegativeRate
	averageWrongRate = totalWrongRate/times
	averageFalsePositiveRate = totalFalsePositiveRate/times
	averageFalseNegativeRate = totalFalseNegativeRate/times
	print times, " times of NBayes Test, average wrong rate is: ", averageWrongRate, "average FalsePositive: ", averageFalsePositiveRate, "average FalseNegative: ", averageFalseNegativeRate
	return 0

if __name__ == '__main__':
	listOfFiles = getAllFiles("./email")
	data, labelList = parseFiles(listOfFiles)
	vocabList = createVocabList(data)
	dataMatrix = words2Bag(data, vocabList)
	testInIteration(1000, 0.2, dataMatrix, labelList, listOfFiles, [1, ])




