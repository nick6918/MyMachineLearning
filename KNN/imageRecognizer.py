from kNeibohood import kNearestNeighbor
from basicFunction import autoNorm
from os import listdir
from numpy import zeros

def image2vector(path, imageFile):
	returnVector = zeros((1024, 1))
	fr = open(path+imageFile)
	for i in range(32):
		line = fr.readline()
		for j in range(32):
			returnVector[32*i+j, 0] = int(line[j])
	return returnVector

def getData(path):
	trainingList = listdir(path)
	trainingListLen = len(trainingList)
	#print trainingListLen
	allData = zeros((1024, trainingListLen))
	label = zeros((trainingListLen, 1))
	for i in range(trainingListLen):
		currentData = image2vector(path, trainingList[i])
		allData[..., i:i+1] = image2vector(path, trainingList[i])
		label[i, 0] = int(trainingList[i].split("_")[0])
	return allData, label

if __name__ == '__main__':
	trainingdata, traininglabel = getData("./digits/trainingDigits/")
	count = 0
	wrong = 0
	for image in listdir("./digits/testDigits/"):
		testdata = image2vector("./digits/testDigits/", image)
		testlabel = int(image.split("_")[0])
		result = kNearestNeighbor(testdata, trainingdata, traininglabel, k=3)
		print count, "th picture, correct: ", testlabel, "learned: ", result
		if testlabel != result:
			wrong += 1
		count += 1
	print "total: ", count, "wrong: ", wrong, "rate: ", wrong*1.0/count
	
