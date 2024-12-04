import cv2
import numpy
import matplotlib.pyplot as plt
import os

def conv(img, conv_filter):
    return feature_maps

def main():
    #Step 1: Input
    folder_path = ""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    images = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        images.append(img)

    #Step 2: Feature detector
    kernel = numpy.zeros((5,3,3))

        #1: Vertical Edge Detection Kernel
    kernel[0, :, :] = numpy.array([[[-1, 0, 1], 
                                    [-1, 0, 1],
                                    [-1, 0, 1]]])
        #2: Horizontal Edge Detection Kernel
    kernel[1, :, :] = numpy.array([[[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]]])
        #3: Sharpening Kernel
    kernel[2, :, :] = numpy.array([[[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]]])
        #4: Sobel Kernel (Horizontal)
    kernel[3, :, :] = numpy.array([[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]])
        #5: Sobel Kernel (Vertical)
    kernel[4, :, :] = numpy.array([[[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]]])
    
    #Step 3: Convolutional layer

if __name__ == "__main__":
    main()