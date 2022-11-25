from doctr.io import DocumentFile
from doctr.models import ocr_predictor, from_hub
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = 'pictures/decathlon.png'

model = ocr_predictor(pretrained=True, det_arch='db_mobilenet_v3_large')
single_img_doc = DocumentFile.from_images(image)
result = model(single_img_doc)

# result.show(single_img_doc)

json_output = result.export()

# with open('json_data.json', 'w') as outfile:
#     json.dump(json_output, outfile)

blocks = json_output["pages"][0]["blocks"]
segments = []
segment = ""
previous_mean = -1
previous_std = -1
for block in blocks:
    lines = block["lines"]
    current_heights = []
    for line in lines:
        words = line["words"]
        y1 = line["geometry"][0][1]
        y2 = line["geometry"][1][1]
        current_segment = ""
        current_heights.append(round(y2 - y1, 3))
        for word in words:
            value = word['value']
            current_segment += f"{value} "
        segment += f"{current_segment}"
    if previous_mean == -1:
        previous_mean = round(np.mean(current_heights), 3)
        previous_std = round(np.std(current_heights), 3)
    else:
        current_mean = round(np.mean(current_heights), 3)
        print(f"segment: {segment}, previous_mean: {previous_mean}, current_mean: {current_mean}, between: {previous_mean - previous_std} and {previous_mean + previous_std}")
        if current_mean < (previous_mean - previous_std) or current_mean > (previous_mean + previous_std):
            segments.append(segment)
            segment = ""
            previous_mean = current_mean
            previous_std = round(np.std(current_heights), 3)
print(segments)

# code based from https://felipemeganha.medium.com/projection-histogram-of-image-using-python-and-opencv-7fd81cadfc23
        
img = cv2.imread(image)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
gray = cv2.medianBlur(gray,5)

thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

kernel = np.ones((5,5), np.uint8)
thresh = cv2.dilate(thresh, kernel ,iterations = 2)
thresh = cv2.erode(thresh, kernel, iterations = 2)

# Find the contours
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if h > 10:    
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# Width and heigth the image
height, width = thresh.shape
horizontal_px = np.sum(thresh,axis=1)
# Sum the value lines 
# vertical_px = np.sum(thresh, axis=0)
# Normalize
normalize = horizontal_px/255
# create a black image with zeros 
blankImage = np.zeros_like(thresh)
# Make the projection histogram
for idx, value in enumerate(normalize):
    cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)
rotated = cv2.rotate(blankImage, cv2.ROTATE_90_CLOCKWISE)    
cleaned = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
plt.imshow(cleaned)
plt.savefig("projection.png")



