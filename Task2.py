from cv2 import imread, imwrite, cvtColor, COLOR_GRAY2RGB, putText, FONT_HERSHEY_SIMPLEX, LINE_AA
import numpy as np
import math as math
from Utilities import sharpen, blur
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


imageALocation = './point.jpg'
imageALocation2 = './point_HQ.jpg'
imageBLocation = './segment.jpg'
PD_R_kernel3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
PD_R_kernel5 = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,8,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
PD_threshold = [747, 1274, 1815] #45,581,10


def writeImage(img, outputFileName):
	imwrite('output/'+outputFileName+'.jpg', img)
	return 1


def main():
	print("2 A: ")
	# Task 2A - 1 - Low Quality Image
	imgA = imread(imageALocation, 0)
	imgALength, imgAWidth = imgA.shape
	resultImg = np.zeros((imgALength, imgAWidth))
	DP_cnt = 0
	pts = []
	for i in range(1, imgALength-1):
		for j in range(1, imgAWidth-1):
			S = imgA[i-1,j-1] * PD_R_kernel3[0,0] + imgA[i-1,j] * PD_R_kernel3[0,1] + imgA[i-1,j+1] * PD_R_kernel3[0,2]
			+ imgA[i,j-1] * PD_R_kernel3[1,0] + imgA[i,j] * PD_R_kernel3[1,1] + imgA[i,j+1] * PD_R_kernel3[1,2]
			+ imgA[i+1,j-1] * PD_R_kernel3[2,0] + imgA[i+1,j] * PD_R_kernel3[2,1] + imgA[i+1,j+1] * PD_R_kernel3[2,2]
			if (abs(S)>PD_threshold[0]):
				DP_cnt = DP_cnt + 1
				resultImg[i,j]=255
				pts.append((i, j))
	writeImage(resultImg, 'Task2A_1')
	print(DP_cnt)
	print("The points detected are at : "+str(pts))
	print()
	# Task 2A - 2 - High Quality Image
	imgA = imread(imageALocation2, 0)
	imgALength, imgAWidth = imgA.shape
	resultImg = np.zeros((imgALength, imgAWidth))
	DP_cnt = 0
	pts = []
	for i in range(2, imgALength-2):
		for j in range(2, imgAWidth-2):
			S = imgA[i-2,j-2] * PD_R_kernel5[0,0] + imgA[i-2,j-1] * PD_R_kernel5[0,1] + imgA[i-2,j] * PD_R_kernel5[0,2] + imgA[i-2,j+1] * PD_R_kernel5[0,3] + imgA[i-2,j+2] * PD_R_kernel5[0,4]
			+ imgA[i-1,j-2] * PD_R_kernel5[1,0] + imgA[i-1,j-1] * PD_R_kernel5[1,1] + imgA[i-1,j] * PD_R_kernel5[1,2] + imgA[i-1,j+1] * PD_R_kernel5[1,3] + imgA[i-1,j+2] * PD_R_kernel5[1,4]
			+ imgA[i,j-2] * PD_R_kernel5[2,0] + imgA[i,j-1] * PD_R_kernel5[2,1] + imgA[i,j] * PD_R_kernel5[2,2] + imgA[i,j+1] * PD_R_kernel5[2,3] + imgA[i,j+2] * PD_R_kernel5[2,4]
			+ imgA[i+1,j-2] * PD_R_kernel5[3,0] + imgA[i+1,j-1] * PD_R_kernel5[3,1] + imgA[i+1,j] * PD_R_kernel5[3,2] + imgA[i+1,j+1] * PD_R_kernel5[3,3] + imgA[i+1,j+2] * PD_R_kernel5[3,4]
			+ imgA[i+2,j-2] * PD_R_kernel5[4,0] + imgA[i+2,j-1] * PD_R_kernel5[4,1] + imgA[i+2,j] * PD_R_kernel5[4,2] + imgA[i+2,j+1] * PD_R_kernel5[4,3] + imgA[i+2,j+2] * PD_R_kernel5[4,4]
			if (abs(S)>PD_threshold[1]):
				DP_cnt = DP_cnt + 1
				resultImg[i,j]=255
				pts.append((i, j))
	writeImage(resultImg, 'Task2A_2')
	print(DP_cnt)
	print("The points detected are at : "+str(pts))
	print()
	# Task 2A - 3 - High Quality Image with Image Sharpening
	imgA = imread(imageALocation2, 0)
	imgA = np.pad(imgA, [(2, 2), (2, 2)], mode='constant', constant_values=0)
	imgA = sharpen(imgA)
	imgALength, imgAWidth = imgA.shape
	resultImg = np.zeros((imgALength, imgAWidth))
	DP_cnt = 0
	pts = []
	for i in range(2, imgALength-2):
		for j in range(2, imgAWidth-2):
			S = imgA[i-2,j-2] * PD_R_kernel5[0,0] + imgA[i-2,j-1] * PD_R_kernel5[0,1] + imgA[i-2,j] * PD_R_kernel5[0,2] + imgA[i-2,j+1] * PD_R_kernel5[0,3] + imgA[i-2,j+2] * PD_R_kernel5[0,4]
			+ imgA[i-1,j-2] * PD_R_kernel5[1,0] + imgA[i-1,j-1] * PD_R_kernel5[1,1] + imgA[i-1,j] * PD_R_kernel5[1,2] + imgA[i-1,j+1] * PD_R_kernel5[1,3] + imgA[i-1,j+2] * PD_R_kernel5[1,4]
			+ imgA[i,j-2] * PD_R_kernel5[2,0] + imgA[i,j-1] * PD_R_kernel5[2,1] + imgA[i,j] * PD_R_kernel5[2,2] + imgA[i,j+1] * PD_R_kernel5[2,3] + imgA[i,j+2] * PD_R_kernel5[2,4]
			+ imgA[i+1,j-2] * PD_R_kernel5[3,0] + imgA[i+1,j-1] * PD_R_kernel5[3,1] + imgA[i+1,j] * PD_R_kernel5[3,2] + imgA[i+1,j+1] * PD_R_kernel5[3,3] + imgA[i+1,j+2] * PD_R_kernel5[3,4]
			+ imgA[i+2,j-2] * PD_R_kernel5[4,0] + imgA[i+2,j-1] * PD_R_kernel5[4,1] + imgA[i+2,j] * PD_R_kernel5[4,2] + imgA[i+2,j+1] * PD_R_kernel5[4,3] + imgA[i+2,j+2] * PD_R_kernel5[4,4]
			if (abs(S)>PD_threshold[2]):
				DP_cnt = DP_cnt + 1
				resultImg[i,j]=255
				pts.append((i, j))
	for pt in pts:
		if (pt==(254,447)):
			putText(resultImg, str((447,254)), (435,240), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=LINE_AA)
	writeImage(resultImg, 'Task2A_3')
	print(DP_cnt)
	print("The points detected are at : "+str(pts))
	print()
	print()

	
	# Task 2B
	print("2 B: ")
	imgB = imread(imageBLocation, 0)
	resultImg = imgB.copy()
	imgBLength, imgBWidth = imgB.shape
	unique, counts = np.unique(imgB, return_counts=True)
	fig, ax = plt.subplots()
	ax.bar(unique, counts, 1)
	#plt.axis([0, 256, 0, 80000])
	plt.xlabel('Pixel Values')
	plt.ylabel('Count')
	plt.title('Histogram - Pixel Values VS Count')
	fig.savefig('output/Task2B_Histogram.png')
	S_threshold = 200
	for i in range(0, imgBLength):
		for j in range(0, imgBWidth):
			if (imgB[i,j]<S_threshold):
				resultImg[i,j]=0
	writeImage(resultImg, 'Task2B_Final_withIndiBoxes')
	minX=None; maxX=None
	minY=None; maxY=None
	x=[]
	y=[]
	for i in range(0, imgBLength):
		for j in range(0, imgBWidth):
			if (imgB[i,j]>S_threshold):
				x.append(j)
				y.append(i)
	x=np.array(x)
	y=np.array(y)
	minX = np.amin(x)
	maxX = np.amax(x)
	minY = np.amin(y)
	maxY = np.amax(y)
	pts = [(minX, minY),(maxX, minY),(maxX, maxY),(minX, maxY)]
	print(pts)
	resultImg[minY:maxY, minX] = 255
	resultImg[minY:maxY, maxX] = 255
	resultImg[minY, minX:maxX] = 255
	resultImg[maxY, minX:maxX] = 255
	putText(resultImg,str((minX, minY)), (minX, minY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	putText(resultImg,str((maxX, minY)), (maxX, minY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	putText(resultImg,str((maxX, maxY)), (maxX, maxY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	putText(resultImg,str((minX, maxY)), (minX, maxY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	writeImage(resultImg, 'Task2B_Final')
	resultImgColor = cvtColor(imgB, COLOR_GRAY2RGB)
	resultImgColor[minY:maxY, minX, 0] = 0; resultImgColor[minY:maxY, minX, 1] = 0; resultImgColor[minY:maxY, minX, 2] = 255
	resultImgColor[minY:maxY, maxX, 0] = 0; resultImgColor[minY:maxY, maxX, 1] = 0; resultImgColor[minY:maxY, maxX, 2] = 255
	resultImgColor[minY, minX:maxX, 0] = 0; resultImgColor[minY, minX:maxX, 1] = 0; resultImgColor[minY, minX:maxX, 2] = 255
	resultImgColor[maxY, minX:maxX, 0] = 0; resultImgColor[maxY, minX:maxX, 1] = 0; resultImgColor[maxY, minX:maxX, 2] = 255
	putText(resultImgColor,str((minX, minY)), (minX, minY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	putText(resultImgColor,str((maxX, minY)), (maxX, minY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	putText(resultImgColor,str((maxX, maxY)), (maxX, maxY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	putText(resultImgColor,str((minX, maxY)), (minX, maxY), FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=LINE_AA)
	writeImage(resultImgColor, 'Task2B_FinalColor')


main()