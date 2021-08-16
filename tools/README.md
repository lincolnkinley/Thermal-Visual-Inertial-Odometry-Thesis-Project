# Thermal-Visual-Inertial-Odometry-Thesis-Project

This directory includes non ros specific tools. 

boson.py will use the Boson sdk to configure the Boson to be externally triggered. You will need the boson sdk to run this program
lepton_record.py was an early iteration of data collection from the Lepton camera. It was eventually replaced with a much more advanced hardware level driver.
process_16.py tests many different contrasting methods for 16-bit images. It will output some graphics showing the results.

You will need OpenCV 3.4 to run boson.py and process_16.py. Newer or later versions may also work. You will also need scipy to run process_16.py
