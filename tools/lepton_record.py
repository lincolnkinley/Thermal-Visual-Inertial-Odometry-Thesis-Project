#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import datetime

parser = argparse.ArgumentParser()

# On my computer this is /dev/video3 but it may be different on you computer. 
# If you are unsure, check /dev/video* with the lepton plugged in, and then unplug it and check again. 
# Typically there are two file names for the camera, one should work and the other shouldn't. Try both.
parser.add_argument("camera", help="File name of the camera to capture.", type=str)

parser.add_argument("bits", help="Number of bits the camera will use to store images. Should be 8 or 16.", type=int)

# Example usage. This will show /dev/video3 with 16 bit images.
# python3 lepton_record.py /dev/video3 16

args = parser.parse_args()

def main():
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open camera " + args.camera)
        exit(1)

    bits = 8
    if args.bits == 8:
        print("Capturing 8 bit AGC data")
    elif args.bits == 16:
        print("Capturing 16 bit data")
        bits = 16
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    else:
        print("Invalid bits, defaulting to capturing 16 bit data")
        bits = 16
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    
    print("Press ESC to exit, or SPACE to save image.");

    # Loop repeats reading the camera until the camera fails to read or the user exits with ESC
    while(True):
        ret, img = cap.read()
        if ret == False:
            print("Failed to read image!")
            break
            
        # For some reason, the Lepton captures two extra rows of black and white pixles when it is set to 16 bit. 
        # I imagine there is a reason why, but this works as a quick fix
        if(bits == 16):
            img = img[:-2, :]
        
        cv2.imshow("lepton", img)
    
        key_pressed = cv2.waitKey(5)
        if key_pressed == 27:
            print("Stopping...")
            break
            
        elif key_pressed == 32:
            print("Saving image...")
            filename = str(datetime.datetime.now()) + ".png"
            filename = filename.replace(" ", "")
            cv2.imwrite(filename, img);
            print(filename + " has been saved.")
            
            # Double check to ensure the saved image is the correct type. The -1 lets OpenCv determine the type
            check_img = cv2.imread(filename, -1)
            if(img.dtype != check_img.dtype):
                print("Warning! Saved image datatype does not match.\nCaptured image: " + str(img.dtype) + "\nSaved image: " + str(check_img.dtype))
            
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
