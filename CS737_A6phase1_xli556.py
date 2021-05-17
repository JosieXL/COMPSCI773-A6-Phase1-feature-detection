from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth

import numpy as np


# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(r_pixel_array,g_pixel_array,b_pixel_array,image_width,image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):

    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage

# this following two functions are used to compute the sobel filter
def computeVerticalEdgesSobel(pixel_array, image_width, image_height):
    Matrix = np.ones((image_height,image_width))
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            Matrix[0][j] = 0.000
            Matrix[-1][j] = 0.000
            Matrix[i][0] = 0.000
            Matrix[i][-1] = 0.000
            
            if (Matrix[i][j]==1.000):
                Matrix[i][j] = (-1)*pixel_array[i-1][j-1] + (-2)*pixel_array[i][j-1] + (-1)*pixel_array[i+1][j-1] + 0*pixel_array[i-1][j] + 0*pixel_array[i][j] + 0*pixel_array[i+1][j] + (1)*pixel_array[i+1][j+1] + (2)*pixel_array[i][j+1] + (1)*pixel_array[i-1][j+1]
    return Matrix

def computeHorizontalEdgesSobel(pixel_array, image_width, image_height):
    Matrix = np.ones((image_height,image_width))
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            Matrix[0][j] = 0.000
            Matrix[-1][j] = 0.000
            Matrix[i][0] = 0.000
            Matrix[i][-1] = 0.000
            
            if (Matrix[i][j]==1.000):
                Matrix[i][j] = (-1)*pixel_array[i+1][j-1] + (-2)*pixel_array[i+1][j] + (-1)*pixel_array[i+1][j+1] + (0)*pixel_array[i][j-1] + (0)*pixel_array[i][j] + (0)*pixel_array[i][j+1] + (1)*pixel_array[i-1][j-1] + (2)*pixel_array[i-1][j] + (1)*pixel_array[i-1][j+1]
    return Matrix

# function for sobel filter and outputs Ix2, Iy2 and IxIy
def SobelDerivativeFilter(pixel_array, image_width, image_height):
    
    # sobel kernel
    #horizontal_kernel_Iy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #vertical_kernel_Ix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    Ix = computeVerticalEdgesSobel(pixel_array, image_width, image_height)
    Iy = computeHorizontalEdgesSobel(pixel_array, image_width, image_height)
    
    # if type is np.matrix, then A*B means matrix multiplication
    # if type is np.ndarray, then A*B means Hadamard product
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy 

    #type of Ix2, Iy2 and IxIy are <class 'numpy.ndarray'>
    return Ix2, Iy2, IxIy

# take a kernel_size and a sigma to create the gaussian kernel
def create_gauss_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma == 0:
        sigma = ((kernel_size - 1)*0.5-1)*0.3+0.8
    for i in range(kernel_size):
        for j in range(kernel_size):
            x,y = i-center, j-center
            kernel[i,j] = 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*(sigma**2)))         
    kernel = kernel/np.sum(kernel)
    return kernel

# function for calculate Gaussian Filter
def GaussianFilter(pixel_array, image_width, image_height, kernel_size, sigma):
    a = int(0.5*(kernel_size-1))
    kernel = create_gauss_kernel(kernel_size,sigma)
    gaussian = np.zeros((image_height,image_width))
    paddingArr = np.pad(pixel_array, pad_width=a, mode='constant', constant_values=0.0)
    for i in range(len(gaussian)):
        for j in range(len(gaussian[i])):
            gaussian[i][j] = np.sum(paddingArr[i:(i+kernel_size), j:(j+kernel_size)]* kernel[:,None])
    return gaussian


# if a matrix is a squared matrix, then can apply following implementation of cornerness score function C
# takes a square n*n matrix M, calculate its determinant
#def det(M):
#    return np.linalg.det(M)
# takes a square n*n matrix M, calculate its trace
#def trace(M):
#    return np.trace(M)
# Cornerness score function C = det(M) - a*trace(M)^2 where Harris constant a=0.04 here
#def CornernessScore(M, a):
#    C = det(M) - a*trace(M)*trace(M)
#return C

# if a matrix is not a squared matrix, then do following with g(Ix2) = gx, g(Iy2) = gy and g(IxIy) = gxy
def CornernessScore(gx, gy, gxy, a=0.04):
    C = gx * gy - gxy**2 - a*((gx + gy)**2)
    return C

# this function is to check if a corner is negative, if it is then changed to 0.0, otherwise keeps the same
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    thresholded = np.array(IPUtils.createInitializedGreyscalePixelArray(image_width, image_height), dtype=np.int64)
    for i in range(len(thresholded)):
        for j in range(len(thresholded[i])):
            if pixel_array[i][j] < threshold_value:
                thresholded[i][j] = 0.0
            else:
                thresholded[i][j] = pixel_array[i][j]
    return thresholded

# function to compute the first 1000 strongest corner and store the coordinates as tuples inside a list
def computeFirst1000StrongestCornerTupleList(pixel_array, image_width, image_height):
    # As append into a numpy.ndarray is much slower than append into a list, so I create an empty list here
    Corner_tuple_list = list()
    pre = np.pad(pixel_array, pad_width=1, mode='constant', constant_values=0.00)
    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[i])):
            if (pixel_array[i][j] > pre[i][j]) and (pixel_array[i][j] > pre[i+1][j]) and (pixel_array[i][j] > pre[i+2][j]) and (pixel_array[i][j] > pre[i][j+1]) and (pixel_array[i][j] > pre[i+2][j+1]) and (pixel_array[i][j] > pre[i+2][j+2]) and (pixel_array[i][j] > pre[i+1][j+2]) and (pixel_array[i][j] > pre[i][j+2]):
                Corner_tuple_list.append((j, i, pixel_array[i][j]))
    Corner_tuple_list.sort(key=lambda tup: tup[2], reverse=True)
    one_thousand_Corner_tuple_list = Corner_tuple_list[:1000]
    oneThousandStrongestCorner = list(dict.fromkeys(one_thousand_Corner_tuple_list))
    return oneThousandStrongestCorner


# This is our code skeleton that performs the stitching
def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"
    
    (image_width, image_height, px_array_left_original)  = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)
    
    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)
    
    # type of px_array_left and px_array_right are <class 'list'>
    # convert them into numpy.array to make it faster
    px_array_left = np.array(px_array_left)
    px_array_right = np.array(px_array_right)

    # the variable which end with 'L' is for px_array_left while end with 'R' is for px_array_right

    Ix2L, Iy2L, IxIyL = SobelDerivativeFilter(px_array_left, image_width, image_height)
    Ix2R, Iy2R, IxIyR = SobelDerivativeFilter(px_array_right, image_width, image_height)

    kernel_size = 5 # 5x5 Gaussian Window - must be odd number
    sigma = 1 # can be calculated by sigma = 0.25*(kernel_size-1)
    
    gxL = GaussianFilter(Ix2L, image_width, image_height, kernel_size, sigma)
    gyL = GaussianFilter(Iy2L, image_width, image_height, kernel_size, sigma)
    gxyL = GaussianFilter(IxIyL, image_width, image_height, kernel_size, sigma)

    gxR = GaussianFilter(Ix2R, image_width, image_height, kernel_size, sigma)
    gyR = GaussianFilter(Iy2R, image_width, image_height, kernel_size, sigma)
    gxyR = GaussianFilter(IxIyR, image_width, image_height, kernel_size, sigma)

    C_arrL = CornernessScore(gxL, gyL, gxyL, 0.04)
    C_arrR = CornernessScore(gxR, gyR, gxyR, 0.04)

    C_after_thresholdL = computeThresholdGE(C_arrL, 0.0, image_width, image_height)
    C_after_thresholdR = computeThresholdGE(C_arrR, 0.0, image_width, image_height)
    
    oneThousandStrongestCornerL = computeFirst1000StrongestCornerTupleList(C_after_thresholdL, image_width, image_height)
    oneThousandStrongestCornerR = computeFirst1000StrongestCornerTupleList(C_after_thresholdR, image_width, image_height)
    

    # some visualizations
    
    fig1, axs1 = pyplot.subplots(1, 2)

    axs1[0].set_title('Harris response left overlaid on orig image')
    axs1[1].set_title('Harris response right overlaid on orig image')
    axs1[0].imshow(px_array_left, cmap='gray')
    axs1[1].imshow(px_array_right, cmap='gray')

    
    # plot a red point in the center of each image
    for i in range(len(oneThousandStrongestCornerL)):
        circleL = Circle((oneThousandStrongestCornerL[i][0], oneThousandStrongestCornerL[i][1]), 1, color='r')
        axs1[0].add_patch(circleL)
    
    for i in range(len(oneThousandStrongestCornerR)):
        circleR = Circle((oneThousandStrongestCornerR[i][0], oneThousandStrongestCornerR[i][1]), 1, color='r')
        axs1[1].add_patch(circleR)

    pyplot.show()

    # a combined image including a red matching line as a connection patch artist (from matplotlib\)

    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)

    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")

    pointA = (image_width/2, image_height/2)
    pointB = (3*image_width/2, image_height/2)
    connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
    ax.add_artist(connection)

    pyplot.show()



if __name__ == "__main__":
    main()

