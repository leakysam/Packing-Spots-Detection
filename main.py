import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    """Compute the absolute difference in average intensity between two images."""
    return np.abs(np.mean(im1) - np.mean(im2))

# Load resources
mask_path = '/Volumes/mac/School/Parking spots/mask_1920_1080.png'
video_path = '/Volumes/mac/School/Parking spots/data/parking_1920_1080_loop.mp4'
mask = cv2.imread(mask_path, 0)
if mask is None:
    raise FileNotFoundError(f"Mask file not found at {mask_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Failed to open video file at {video_path}")

# Identify parking spots from mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Initialize parking lot status tracking
spots_status = [None] * len(spots)
diffs = [None] * len(spots)
previous_frame = None
frame_nmr = 0
step = 30  # Process every 'step' frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0:
        if previous_frame is not None:
            # Compute differences in parking spot regions
            for i, (x1, y1, w, h) in enumerate(spots):
                spot_crop_current = frame[y1:y1+h, x1:x1+w]
                spot_crop_previous = previous_frame[y1:y1+h, x1:x1+w]
                diffs[i] = calc_diff(spot_crop_current, spot_crop_previous)

            # Print and plot the sorted normalized differences
            sorted_indices = np.argsort(diffs)[::-1]
            print([diffs[i] for i in sorted_indices])
            plt.figure()
            plt.hist([diffs[i] / max(diffs) for i in sorted_indices])
            plt.title("Histogram of Differences")
            

        # Update parking spots status
        if previous_frame is None:
            indices_to_check = range(len(spots))
        else:
            threshold = 0.4
            indices_to_check = [i for i in sorted_indices if diffs[i] / max(diffs) > threshold]
        
        for i in indices_to_check:
            x1, y1, w, h = spots[i]
            spot_crop = frame[y1:y1+h, x1:x1+w]
            spots_status[i] = empty_or_not(spot_crop)

        previous_frame = frame.copy()

    # Update frame visualization based on spot statuses
    for i, (x1, y1, w, h) in enumerate(spots):
        color = (0, 255, 0) if spots_status[i] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots {spots_status.count(True)} / {len(spots)}', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
