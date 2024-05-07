import cv2
import matplotlib.pyplot as plt
import numpy as np

def calc_diff(im1,im2):   # function to compute how similar or differenct the arking spots are
    return np.abs(np.mean(im1) - np.mean(im2))

mask = '/Volumes/mac/School/Parking spots/mask_1920_1080.png'
video_path ='/Volumes/mac/School/Parking spots/data/parking_1920_1080_loop.mp4'

from util import get_parking_spots_bboxes, empty_or_not


mask = cv2.imread(mask,0)
cap = cv2.VideoCapture(video_path)

#find location of packing spots
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)#no connection between the different groups(graph theory) (CV_32S, represents discrete data for 32bit and used because it's an integer)

spots = get_parking_spots_bboxes(connected_components)#Output parking spots

spots_status = [None for j in spots]
diffs = [None for j in spots]
previous_frame = None # save previous frame

frame_nmr = 0
ret = True
step =30 # how often we classify the packing lot then update the frames after.
while ret:
    ret, frame = cap.read()
    
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 +h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 +h, x1:x1 + w, :]) # compute difference between parking spot in the previous frame
        print([diffs[j] for j in np.argsort(diffs)][::-1])# print(print(diffs)) the numbers to show it's working  showing how different the parking lot is from the different frame  (get the biggest value first by sorting it)
        plt.figure()
        plt.hist([diffs[j]/ np.amax(diffs) for j in np.argsort(diffs)][::-1])# draw the histogrm showing the differences in values(plt.show())
        if frame_nmr == 300:# draw histograms for all the differences in the parking spots if small value nothing is taking place if a bigger value something is taking place
            plt.show()
    
    #update value of parking spot status
    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [ j for j in np.argsort(diffs) if diffs[j]/ np.amax(diffs) > 0.4 ]
            
        for spot_indx in arr_:    #for spot_indx, spot in enumerate(spots):(was here before)
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 +h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status
    
    if frame_nmr % step == 0:
        previous_frame = frame.copy() # save value after each iteration of the loop function
            
      # draw based on spot status updated above  taking most recent value    
    for spot_indx, spot in enumerate(spots):  
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]     
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1),(x1+w, y1 + h), (0, 255, 0),2)#display color green on the frames for available parking spots
        else:
            frame = cv2.rectangle(frame, (x1, y1),(x1+w, y1 + h), (0, 0, 255),2)#red color for parking spot not available
    
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF ==ord('q'):
        break
    frame_nmr += 1 # increment variable 

cap.release()
cv2.destroyAllWindows()