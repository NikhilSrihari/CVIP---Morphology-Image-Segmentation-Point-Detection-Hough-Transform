from cv2 import imread, imwrite, line as l
import numpy as np
import math as math
from Utilities import sharpen, blur, edgeDetector


imageLocation = './hough.jpg'
rImgThreshold = 75
gImgThreshold = 0
bImgThreshold = 75
houghTransformApplyThreshold = 50
houghTransformLineThreshold_numOfLines_vertical = 25
houghTransformLineThreshold_numOfLines_slant = 48
thetaOffset = -90


def writeImage(img, outputFileName):
	imwrite('output/Task3_'+outputFileName+'.jpg', img)
	return 1


def thresholdImg(img0, threshold):
	img = img0.copy()
	for i in range(0, len(img)):
		for j in range(0, len(img[0])):
			if (img[i,j]<threshold):
				img[i,j] = 0
	return img


def createAccumulatorImg(img, imgLength, imgWidth, imgDiagLen=None):
	if (imgDiagLen==None):
		imgDiagLen = math.ceil(math.sqrt(imgLength**2 + imgWidth**2))
	accImgLength = 2*imgDiagLen+1
	accImgWidth = 181
	accImg = np.zeros((accImgLength, accImgWidth))
	#2,2 (1639, 361)
	for i in range(0, imgLength):
		y = (imgLength-1)-i
		for j in range(0, imgWidth):
			x = j
			if(img[i,j]>houghTransformApplyThreshold):
				for l in range(0, accImgWidth):
					o = math.radians(thetaOffset+l)
					p = x*math.cos(o) + y*math.sin(o)
					k = int( imgDiagLen - math.floor(p) )
					accImg[k, l] = accImg[k, l] + 1
	return accImg


def detectLines(baseImg, img, imgLength, imgWidth, imgDiagLen=None, lineType=None):
	print(lineType+" : ")
	accImg = createAccumulatorImg(img, imgLength, imgWidth, imgDiagLen)
	accImgLength, accImgWidth = accImg.shape
	writeImage(accImg, 'accImg_'+lineType)
	if (lineType=='VERTICAL'):
		houghTransformLineThreshold_numOfLines = houghTransformLineThreshold_numOfLines_vertical
		for j in range(0, accImgWidth):
			if (j<85 or j>95):
				accImg[:, j] = 0
	elif (lineType=='SLANT'):
		houghTransformLineThreshold_numOfLines = houghTransformLineThreshold_numOfLines_slant
		for j in range(0, accImgWidth):
			if (j<=126 or j>=128):
				accImg[:, j] = 0
	writeImage(accImg, 'accImg_'+lineType+'_LineOrientationSpecific')
	lines = []
	accImgSorted = np.sort(accImg.flatten())
	houghTransformLineThreshold_Top = accImgSorted[(accImgLength*accImgWidth)-houghTransformLineThreshold_numOfLines]
	for i in range(0, accImgLength):
		for j in range(0, accImgWidth):
			if ( accImg[i,j]>=houghTransformLineThreshold_Top ):
				lines.append( ( (imgDiagLen-i), (thetaOffset+j) ) )
	print("	Number of lines detected : "+str(len(lines)))
	print("	Line (p,o) values : "+str(lines))
	rawLinesImg = np.zeros((imgLength, imgWidth))
	#lines = [(394, 37)]
	for line in lines:
		p = line[0]
		o = line[1]
		o = math.radians(o)
		cosO = math.cos(o)
		sinO = math.sin(o)
		if (lineType=='VERTICAL'):
			y1 = 0
			x1 = int( (p-y1*sinO)/cosO )
			y2 = (imgLength-1)
			x2 = int( (p-y2*sinO)/cosO )
			l(rawLinesImg, (x1,(imgLength-1)-y1),(x2,(imgLength-1)-y2),(255,255,255),2)
			l(baseImg, (x1,(imgLength-1)-y1),(x2,(imgLength-1)-y2),(0,0,255),2)
		else:
			y1 = 0
			x1 = int( (p-y1*sinO)/cosO )
			y2 = (imgLength-1)
			x2 = int( (p-y2*sinO)/cosO )
			l(rawLinesImg, (x1,(imgLength-1)-y1),(x2,(imgLength-1)-y2),(255,255,255),2)
			l(baseImg, (x1,(imgLength-1)-y1),(x2,(imgLength-1)-y2),(255,0,0),2)
	return rawLinesImg, baseImg


def main():
	img = imread(imageLocation)
	#(477, 666, 3)
	imgLength, imgWidth, numOfChannels = img.shape
	imgDiagLen = math.ceil(math.sqrt(imgLength**2 + imgWidth**2))
	rImg = thresholdImg(img[:,:,0], rImgThreshold)
	gImg = thresholdImg(img[:,:,1], gImgThreshold)
	bImg = thresholdImg(img[:,:,2], bImgThreshold)
	sobelXRImg, sobelYRImg, finalEdgeDetectedRImg = edgeDetector(rImg)
	writeImage(finalEdgeDetectedRImg, "finalEdgeDetectedRImg")
	sobelXGImg, sobelYGImg, finalEdgeDetectedGImg = edgeDetector(gImg)
	writeImage(finalEdgeDetectedGImg, "finalEdgeDetectedGImg")
	sobelXBImg, sobelYBImg, finalEdgeDetectedBImg = edgeDetector(bImg)
	writeImage(finalEdgeDetectedBImg, "finalEdgeDetectedBImg")
	'''finalEdgeDetectedRImg = imread('./output/Task3_Sol1_finalEdgeDetectedRImg.jpg', 0)
	finalEdgeDetectedGImg = imread('./output/Task3_Sol1_finalEdgeDetectedGImg.jpg', 0)
	finalEdgeDetectedBImg = imread('./output/Task3_Sol1_finalEdgeDetectedBImg.jpg', 0)
	imgLength, imgWidth = finalEdgeDetectedRImg.shape
	imgDiagLen = math.ceil(math.sqrt(imgLength**2 + imgWidth**2))'''
	rawLinesImg_v, baseImg_v = detectLines(img.copy(), finalEdgeDetectedRImg, imgLength, imgWidth, imgDiagLen, 'VERTICAL')
	writeImage(rawLinesImg_v, 'BW_verticalLines')
	writeImage(baseImg_v, 'verticalLines')
	rawLinesImg_s, baseImg_s = detectLines(img.copy(), finalEdgeDetectedBImg, imgLength, imgWidth, imgDiagLen, 'SLANT')
	writeImage(rawLinesImg_s, 'BW_slantLines')
	writeImage(baseImg_s, 'slantLines')


main()