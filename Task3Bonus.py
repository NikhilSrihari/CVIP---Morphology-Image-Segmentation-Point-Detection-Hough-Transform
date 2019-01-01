from cv2 import imread, imwrite, circle as cir
import numpy as np
import math as math
from Utilities import sharpen, blur, edgeDetector


imageLocation = './hough.jpg'
imgThreshold = 0
houghTransformApplyThreshold = 30
radius = 24
houghTransformLineThreshold_numOfCircles = 90


def writeImage(img, outputFileName):
	imwrite('output/Task3_Bonus_'+outputFileName+'.jpg', img)
	return 1


def thresholdImg(img0, threshold):
	img = img0.copy()
	for i in range(0, len(img)):
		for j in range(0, len(img[0])):
			if (img[i,j]<threshold):
				img[i,j] = 0
	return img


def createAccumulatorImg(img, imgLength, imgWidth):
	accImg = np.zeros((imgLength, imgWidth))
	for i in range(0, imgLength):
		y = i
		for j in range(0, imgWidth):
			x = j
			if(img[i,j]>houghTransformApplyThreshold):
				for l in range(0, 360):
					o = math.radians(l)
					a = int(x - radius*math.cos(o))
					b = int(y - radius*math.sin(o))
					if (a>=0 and a<imgWidth and b>=0 and b<imgLength):
						accImg[b, a] += 1
	return accImg


def detectCircles(baseImg, img, imgLength, imgWidth):
	accImg = createAccumulatorImg(img, imgLength, imgWidth)
	accImgLength, accImgWidth = accImg.shape
	writeImage(accImg, 'accImg_circles')
	circleCenters = []
	accImgSorted = np.sort(accImg.flatten())
	houghTransformLineThreshold_Top = accImgSorted[(accImgLength*accImgWidth)-houghTransformLineThreshold_numOfCircles]
	for i in range(0, accImgLength):
		for j in range(0, accImgWidth):
			if ( accImg[i,j]>=houghTransformLineThreshold_Top ):
				circleCenters.append((j,i))
	print("	Number of coins detected : "+str(len(circleCenters)))
	print("	circleCenters (a,b) values : "+str(circleCenters))
	rawImg = np.zeros((imgLength, imgWidth))
	for circleCenter in circleCenters:
		a = circleCenter[0]
		b = circleCenter[1]
		center = (a, b)
		cir(rawImg, center, radius, (255,255,255), 2) 
		cir(baseImg, center, radius, (0,0,255), 2)
	return rawImg, baseImg


def main():
	colorImg = imread(imageLocation)
	img = thresholdImg(colorImg[:,:,1], imgThreshold)
	#img = imread(imageLocation, 0)
	imgLength, imgWidth = img.shape
	#img = thresholdImg(img, imgThreshold)
	sobelXImg, sobelYImg, finalEdgeDetectedImg = edgeDetector(img)
	writeImage(finalEdgeDetectedImg, "finalEdgeDetectedImg_circles")
	'''finalEdgeDetectedImg = imread('./output/Task3_Bonus_finalEdgeDetectedImg_circles.jpg', 0)
	imgLength, imgWidth = finalEdgeDetectedImg.shape'''
	rawCirclesImg, baseCirclesImg = detectCircles(colorImg, finalEdgeDetectedImg, imgLength, imgWidth)
	writeImage(rawCirclesImg, 'BW_rawCirclesImg')
	writeImage(baseCirclesImg, 'C_baseCirclesImg')


main()