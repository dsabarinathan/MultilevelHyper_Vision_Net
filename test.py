# -*- coding: utf-8 -*-

from keras.models import load_model

from numpy import newaxis
import numpy as np
import cv2
import os
import argparse
import time
from scipy.io.matlab.mio import savemat
from coord import CoordinateChannel2D
from model_utils import sum_squared_error, ssim,PSNR
from subpixelupscaling import SubPixelUpscaling,SubpixelConv2D
import tensorflow as tf

IMG_WIDTH = IMG_HEIGHT =128

if __name__ == '__main__':
         
    parser = argparse.ArgumentParser(description='eye-net')
    parser.add_argument("--testImagePath", type=str,dest="test_path" ,help="Path of test Images",default='./test/',action="store")
    args = parser.parse_args()
    

#    model = load_model('./model/model-221.74-val_mse-0.0004--val_ssim--0.9827.hdf5',custom_objects={'sum_squared_error':sum_squared_error,'ssim':ssim,'CoordinateChannel2D':CoordinateChannel2D})
    
    model = load_model('./model/model--BuildResidual_wihtoutcbam_GRAB-input_128--CBAM-attention-122-val_op_loss--79.8241-val_PSNR-34.5207.hdf5',custom_objects={'PSNR':PSNR,'sum_squared_error':sum_squared_error,'ssim':ssim,'CoordinateChannel2D':CoordinateChannel2D,'SubPixelUpscaling':SubPixelUpscaling,'SubpixelConv2D':SubpixelConv2D,'tf':tf})
    output_path = './output_file/'
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    
    testImagePath = args.test_path
    
    fileName = os.listdir(testImagePath)
    X_test = np.zeros((1,IMG_HEIGHT, IMG_WIDTH, 3))
    
    for i in range(len(fileName)):
        
        start_time = time.time()

        imr = cv2.imread(testImagePath+fileName[i])
   
        resized=cv2.resize((imr),(IMG_WIDTH,IMG_HEIGHT))
        
        X_test[0]=resized/255
        
        predicted_test=model.predict(X_test,batch_size=4,verbose=1)

        end_time = time.time()
    
        print('predicted time', end_time-start_time)

        
        cv2.imwrite(output_path+fileName[i].split('_')[0]+'_gt.png',predicted_test[0]*255)

    
        print(i)
        
    print("output files saved in "+output_path)
