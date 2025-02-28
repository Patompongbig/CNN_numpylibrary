import cv2
import numpy
import matplotlib.pyplot as plt
import os

#Function perform convolution for a single filter over an image
def conv_(img, kernel_filter):
    filter_size = kernel_filter.shape[1]
    result = numpy.zeros((img.shape))

    #Loop through the image
    for r in numpy.uint16(numpy.arange(filter_size/2.0,
                                       img.shape[0] - filter_size/2.0 + 1)):
        for c in numpy.uint16(numpy.arange(filter_size/2.0,
                                           img.shape[1] - filter_size/2.0 + 1)):
            current_region = img[r - numpy.uint16(numpy.floor(filter_size/2.0)): r + numpy.uint16(numpy.ceil(filter_size / 2.0)),
                                 c - numpy.uint16(numpy.floor(filter_size/2.0)): c + numpy.uint16(numpy.ceil(filter_size / 2.0))]
            
            #Perform Convolution
            current_result = current_region * kernel_filter
            conv_sum = numpy.sum(current_result)

            result[r, c] = conv_sum

    #Crop the result of the matrix
    final_result = result[numpy.uint16(filter_size/2.0): result.shape[0] - numpy.uint16(filter_size/2.0),
                        numpy.uint16(filter_size/2.0): result.shape[1] - numpy.uint16(filter_size/2.0)]
            
    return final_result


#Function manage multiple filter and Convolutional layer
def conv(img, kernel_filter):
    #CHECK ERROR!!
    #Check number of image channels match the filter depth
    if len(img.shape) > 2 or len(kernel_filter.shape) > 3:
        if img.shape[-1] != kernel_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()

    #Check filter dimension like 3*3, 5*5
    if kernel_filter.shape[1] != kernel_filter.shape[2]:
        print("Error: filter must be a square matrix.")
        sys.exit()

    #Check filter dimension are odd
    if kernel_filter.shape[1] % 2 == 0:
        print("Error: Filter must have an odd size.")
        sys.exit()

    #Create Feature Map
    #Create empty matrix of feature map
    feature_maps = numpy.zeros((img.shape[0] - kernel_filter.shape[1] + 1,
                                img.shape[1] - kernel_filter.shape[1] + 1,
                                kernel_filter.shape[0]))
    
    #Convolving the image
    #Rotate with each filter
    for filter_num in range(kernel_filter.shape[0]):
        print("Filter ", filter_num + 1)
        current_filter = kernel_filter[filter_num, :]

        #Convolution with each filter
        #For RGB image
        if len(current_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], current_filter[:, :, 0])
            for channel_num in range(1, current_filter.shape[-1]):
                conv_map = conv_map + conv_(img[:, :, channel_num],
                                            current_filter[:, :, channel_num])
        
        #For Greyscale image
        else: 
            conv_map = conv_(img, current_filter)

        #Storing the result in the feature map array
        feature_maps[:, :, filter_num] = conv_map

    return feature_maps


#Function loop through each pixel in feature map and return original value if it larger than 0
def relu(feature_map):
    #Duplicate layer
    relu_out = numpy.zeros(feature_map.shape)

    #Apply relu to get only positive value in each pixel
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0, feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])

    return relu_out


#Function divide the feature map into small regions and apply a pooling
def pooling(feature_map, size = 2, stride = 2):
    #Create output with specific size
    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0] - size + 1) / stride),
                            numpy.uint16((feature_map.shape[1] - size + 1) / stride),
                            feature_map.shape[-1]))
    

    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in numpy.arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in numpy.arange(0, feature_map.shape[1] - size + 1, stride):
                pool_out[r2, c2, map_num] = numpy.max(feature_map[r:r+size, c:c+size, map_num])
                c2 = c2 + 1
            r2 = r2 + 1

    return pool_out


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
    feature_map = conv(img, kernel)

    #Step 4: ReLU layer
    feature_map_relu = relu(feature_map)

    #Step 5: Max Pooling layer
    feature_map_relu_pool = pooling(feature_map_relu, 2, 2)


if __name__ == "__main__":
    main()