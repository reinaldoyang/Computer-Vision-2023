import cv2 as cv
import numpy as np

img = cv.imread('StadiumSnap.jpg')

try:
    #open and clean trajectory file
    with open('Trajectory.xyz', 'r') as f:
        trajectory = [line.rstrip('\n') for line in f]
        trajectory = [[float(x) for x in string.split()]for string in trajectory]
except FileNotFoundError:
    print("File does not exist")

try:
    #open and clean parameter file
    with open('Camera Parameter.txt') as f:
        lines = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("File does not exist")

homogeneous_value = 1
for i in trajectory:
    i.append(homogeneous_value)

param = []
for line in lines:
    if '[' in line and ']' in line:
        row = [float(x) for x in line.replace('[', '').replace(']', '').split()]
        param.append(row)

intrinsic_parameter = param[:3]
extrinsic_parameter = param[3:]
print('Intrinsic Parameter :', intrinsic_parameter)
print('Extrinsic Parameter :', extrinsic_parameter)

cam_coordinates = []
#calculate extrinsic value with trajectory coordinates
for i, coord in enumerate(trajectory):
    cam_coordinates.append([])
    for j in extrinsic_parameter:
        cam_coordinates[i].append(np.matmul(coord,j))
        
#calculate image coordinates
img_coordinates = []
for i, coord in enumerate(cam_coordinates):
    img_coordinates.append([])
    for j in intrinsic_parameter:
        img_coordinates[i].append(np.matmul(coord, j))

#divide xz,yz with z
for i in img_coordinates:
    i[0] = i[0]/i[2]
    i[1] = i[1]/i[2]
    i[2] = i[2]/i[2]

#delete last element from each coordinate
for coord in img_coordinates:
    del coord[-1]
    
pts = np.array(img_coordinates, np.int32)
cv.polylines(img, [pts], False, (0, 255, 0), thickness = 2)
cv.imshow("Original Image", img)
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()

filename = 'B10803207.jpg'
cv.imwrite(filename, img)