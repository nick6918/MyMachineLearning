from basicFunction import file2Matrix, createExampleData, createLabeledData, autoNorm
from math import floor

def kNearestNeighbor(vector, data, labels, k=1):
	#require: data and vector has same amount of feature. 
	#require: all data should be normalized first
	distMatrix = (data - vector)**2
	dist = (distMatrix.sum(axis=0))**0.5
	sortedLabelIndices = dist.argsort()
	classCount = {}
	for index in range(k):
		votedLabel = labels[sortedLabelIndices[index], 0]
		classCount[votedLabel] = classCount.get(votedLabel, 0)+1
	maxLabel = None
	maxCount = 0
	for label in classCount.keys():
		if classCount[label] >= maxCount:
			maxLabel = label
			maxCount = classCount[label]
	return maxLabel

def main():
	#data, labels = createLabeledData(10, 5, 4, 0, 10)
	#data, labels = createExampleData()
	data, labels = file2Matrix('datingTestSet.txt', 3)
	trainDataLen = floor(0.9*len(data[0]))
	data = autoNorm(data)
	trainData = data[..., 0:trainDataLen]
	trainLabel = labels[0:trainDataLen]
	testData = data[..., trainDataLen:len(data[0])]
	testLabel = labels[trainDataLen:len(labels)]
	wrongCount = 0
	for index in range(len(testData[0])):
		currentData = testData[..., index:index+1]
		result = kNearestNeighbor(currentData, trainData, trainLabel, k=3)
		correctResult = testLabel[index, 0]
		print index, "th data, correct: ",correctResult, "result: ", result
		if result != correctResult:
			print currentData
			wrongCount += 1
	print "Total :", len(testData[0]), "wrong: ", wrongCount, "rate :", wrongCount*1.0/len(testData[0]) 

if __name__ == '__main__':
	main()



