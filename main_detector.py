# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from cnn import main
import pandas as pd



image = cv2.imread("img.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break

if len(approx)<4:
    print("Kindly make sure edges of paper were not hidden")
    

warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)

thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
    

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
#loop over the digit area candidates
for c in cnts:
 	# compute the bounding box of the contour
 	(x, y, w, h) = cv2.boundingRect(c)
 	# if the contour is sufficiently large, it must be a digit
 	if w >= 15 and (h >= 30 ):
          digitCnts.append(c)
digitCnts = contours.sort_contours(digitCnts,
 	method="left-to-right")[0]


ph=[]

for c in digitCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    ROI = thresh[y-30:y+h+30, x-30:x+w+30]
    ROI = cv2.resize(ROI, (28,28),interpolation = cv2.INTER_AREA)
    ph.append(main(ROI))


#search phone number in directory and print name
ph_str = ''.join(map(str, ph))
ph_str=int(ph_str)
df = pd.read_excel (r'directory.xlsx')
binary=0
for i, row in df.iterrows():
    if ph_str == row['Telephone number']:
        print(row['Name'])
        binary=1
if binary==0:    
    for i in range(len(ph)):
        for j in range(i + 1, len(ph)):

            if ph[i] < ph[j]:
                ph[i], ph[j] = ph[j], ph[i]

    new_list_str = ''.join(map(str, ph))
    print("The given number is not available in the directory. \nStage 3 activated.")
    print("Largest possible number from input = ",new_list_str)
