#!/usr/bin/env python2

import rospy
import pickle
import rosbag
import argparse
import math
from gps.msg import gnss_data

parser = argparse.ArgumentParser(description="Break rosbag into images")
parser.add_argument("bagfile", help="Name of the bagfile used as input")
args = parser.parse_args()



def main():
    print("Opening rosbags...")
    read_bag = rosbag.Bag(args.bagfile, 'r')
    print("rosbags opened!")
    
    bag_size = read_bag.get_message_count("/gnss_data")

    loop = 0
    pct = 0
    
    gnss_data = []

   
    for topic, msg, time, in read_bag.read_messages(topics="/gnss_data"):
        if((100*float(loop))/float(bag_size) > float(pct)):
            print(str(pct) + "% complete")
            pct += 1
        
        loop += 1

        if(topic == "/gnss_data"):
            gnss_data.append([msg.utm_easting, msg.utm_northing])

    pickle.dump( gnss_data, open( "gnss_data_flight_4.p", "wb" ) )

    read_bag.close()
        
	
if __name__ == "__main__":
    main()
        
