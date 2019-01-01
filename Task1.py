from cv2 import imread, imwrite
import numpy as np


imageLocation = './noise.jpg'


def fetchKernel():
	return np.ones((3, 3), np.uint8)


def writeImage(img, outputFileName):
	imwrite('output/'+outputFileName+'.jpg', img)
	return 1


def dilateImg(img, kernel):
	imgLength, imgWidth = img.shape
	resultImg = np.zeros((imgLength-2, imgWidth-2))
	for i in range(1, imgLength-1):
		for j in range(1, imgWidth-1):
			if( (img[i-1,j-1]==255 and kernel[0,0]==1) or (img[i-1,j]==255 and kernel[0,1]==1) or (img[i-1,j+1]==255 and kernel[0,2]==1) 
				or (img[i,j-1]==255 and kernel[1,0]==1) or (img[i,j]==255 and kernel[1,1]==1) or (img[i,j+1]==255 and kernel[1,2]==1) 
				or (img[i+1,j-1]==255 and kernel[2,0]==1) or (img[i+1,j]==255 and kernel[2,1]==1) or (img[i+1,j+1]==255 and kernel[2,2]==1)):
				resultImg[i-1][j-1] = 255
	return resultImg


def erodeImg(img, kernel):
	imgLength, imgWidth = img.shape
	resultImg = np.zeros((imgLength-2, imgWidth-2))
	for i in range(1, imgLength-1):
		for j in range(1, imgWidth-1):
			if( (img[i-1,j-1]==255 and kernel[0,0]==1) and (img[i-1,j]==255 and kernel[0,1]==1) and (img[i-1,j+1]==255 and kernel[0,2]==1) 
				and (img[i,j-1]==255 and kernel[1,0]==1) and (img[i,j]==255 and kernel[1,1]==1) and (img[i,j+1]==255 and kernel[1,2]==1) 
				and (img[i+1,j-1]==255 and kernel[2,0]==1) and (img[i+1,j]==255 and kernel[2,1]==1) and (img[i+1,j+1]==255 and kernel[2,2]==1)):
				resultImg[i-1][j-1] = 255
	return resultImg


def openImg(img, kernel):
    img1 = erodeImg(img, kernel)
    resultImg = dilateImg(img1, kernel)
    return resultImg


def closeImg(img,kernel):
    img1 = dilateImg(img, kernel)
    resultImg = erodeImg(img1,kernel)
    return resultImg


def detectBoundaries(img, kernel):
    erodedImg = erodeImg(img, kernel)
    erodedImg = np.pad(erodedImg, (1, 1), 'constant')
    resultImg = np.asarray(img - erodedImg)
    return resultImg


def main():
	# Task 1A
	kernel = fetchKernel()
	img = imread(imageLocation, 0)
	openImg1 = openImg(img, kernel)
	noiseReduction_resultImg1 = closeImg(openImg1, kernel)
	closeImg1 = closeImg(img, kernel)
	noiseReduction_resultImg2 = openImg(closeImg1, kernel)
	writeImage(noiseReduction_resultImg1, 'res_noise1.jpg')
	writeImage(noiseReduction_resultImg2, 'res_noise2.jpg')
	# Task 1C
	boundaryDetection_resultImg1 = detectBoundaries(noiseReduction_resultImg1, kernel)
	boundaryDetection_resultImg2 = detectBoundaries(noiseReduction_resultImg2, kernel)
	writeImage(boundaryDetection_resultImg1, 'res_bound1.jpg')
	writeImage(boundaryDetection_resultImg2, 'res_bound2.jpg')

main()