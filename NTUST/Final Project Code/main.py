import os
import cv2
import numpy as np

def load_calibration_data(calibration_file_path):
  with open(calibration_file_path, 'r') as file:
    calibration_data = []
    for line in file:
      line = line.strip()
      if line and not line.startswith('#'):
        values = line.split()
        calibration_data.extend([float(value) for value in values])

  calibration_data = np.array(calibration_data)

  K_left = calibration_data[:9].reshape(3, 3)
  K_right = calibration_data[21:30].reshape(3, 3)

  RT_left = calibration_data[9:21].reshape(3, 4)
  RT_right = calibration_data[30:42].reshape(3, 4)

  F_matrix = calibration_data[42:].reshape(3, 3)

  return K_left, K_right, RT_left, RT_right, F_matrix

def draw_circle(image, points):
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 3, (255, 0, 0), 3)

def calculate_brightest_point(left_gray, threshold):
    points_left = []
    for row in range(700):
        max_value_left = np.max(left_gray[row])
        if max_value_left >= threshold:
            left_pixel_indices = np.where(left_gray[row] == max_value_left)[0]
            # if len(left_pixel_indices) > 0 and len(right_pixel_indices) > 0:
            if len(left_pixel_indices) > 0:
                left_pixel = left_pixel_indices[0]
                points_left.append([left_pixel, row])
    return points_left
                

def brightest_point_epiline(epipolar_lines):
    for epipolar_line in epipolar_lines:
        a, b, c = epipolar_line[0]
        brightest_point = None
        max_brightness = 0

        for x in range(right_view.shape[1]):
            y = float(-(a * x + c) / b)
            if y >= 0 and y < right_view.shape[0]:
                y = int(y)
                brightness = right_gray[y, x]
                if brightness > max_brightness:
                    max_brightness = brightness
                    brightest_point = (x, y)
        # if brightest_point is not None:
        brightest_points.append(brightest_point)
        # cv2.circle(right_view, brightest_point, 3, (250, 0, 0), 3)

def draw_epiline(epipolar_lines):
    for epipolar_line in epipolar_lines:
        x0, y0 = map(int, [0, -epipolar_line[0][2] / epipolar_line[0][1]])
        x1, y1 = map(int, [right_view.shape[1] - 1, -(epipolar_line[0][0] * (right_view.shape[1] - 1) + epipolar_line[0][2]) / epipolar_line[0][1]])
        cv2.line(right_view, (x0, y0), (x1, y1), (0, 0, 255), 1)


def triangulate_points(P_left, P_right, left_points, right_points):
    p1 = P_left[0]
    p2 = P_left[1]
    p3 = P_left[2]
    p1_right = P_right[0]
    p2_right = P_right[1]
    p3_right = P_right[2]

    X_matrix = []
    for i in range(len(left_points)):
        u = left_points[i][0]
        v = left_points[i][1]
        u_right = right_points[i][0]
        v_right = right_points[i][1]

        A = np.array([
            u * p3 - p1,
            v * p3 - p2,
            u_right * p3_right - p1_right,
            v_right * p3_right - p2_right
        ])
        _, _, X = np.linalg.svd(A)
        X = X[-1]  # Select the last column of V
        X = X / X[-1]
        X = X[:3]
        X_matrix.append(X)
    return X_matrix
    

def save_file(all_points_3d):
    # Save all 3D points as a single XYZ file
    output_path = os.path.join('SidebySide', 'output.xyz')
    with open(output_path, 'w') as file:
        for points_2d in all_points_3d:
            for point in points_2d:
                file.write(f'{point[0]} {point[1]} {point[2]}\n')

def remove_low_brightness_points(brightest_points, left_points, right_gray, threshold):
    filtered_brightest_points = []
    filtered_left_points = []

    for i in range(len(brightest_points)):
        x, y = brightest_points[i]
        brightness = right_gray[y, x]
        if brightness >= threshold:
            filtered_brightest_points.append(brightest_points[i])
            filtered_left_points.append(left_points[i])

    return filtered_brightest_points, filtered_left_points

#Load calibration data
K_left, K_right, RT_left, RT_right, F_matrix = load_calibration_data('./SidebySide/CalibrationData.txt')
# Build camera projection matrices
P_left = np.matmul(K_left, RT_left)
P_right = np.matmul(K_right, RT_right)

print("K_left\n", K_left)
print("K_right\n", K_right)
print("RT_left\n", RT_left)
print("RT_right\n", RT_right)
print("Fundamental Matrix\n", F_matrix)
# Print the camera projection matrices
print("P_left:\n", P_left)
print("P_right:\n", P_right)

image_path = "./SidebySide/SBS_065.jpg"


# Get image file list
image_dir = 'SidebySide'
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
all_3d_points = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)

    # Load the image
    frame = cv2.imread(image_path)

    if frame is None:
        print(f'Unable to read image file: {image_path}')
        continue

    # Split the frame into left and right views
    height, width, _ = frame.shape
    left_view = frame[:, :width//2]
    right_view = frame[:, width//2:]


    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_view, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_view, cv2.COLOR_BGR2GRAY)

    # points_right = []

    points_left = calculate_brightest_point(left_gray, 40)
    epipolar_lines = cv2.computeCorrespondEpilines(np.array(points_left), 2, F_matrix)

    brightest_points = []  # Use a set instead of a list
    brightest_point_epiline(epipolar_lines)
    filtered_brightest_points, filtered_left_points = remove_low_brightness_points(brightest_points, points_left, right_gray, 100)
    X_matrix = triangulate_points(P_left, P_right, filtered_brightest_points, filtered_brightest_points)
    all_3d_points.append(X_matrix)

save_file(all_3d_points)

# draw_circle(left_view, points_left)
# # draw_circle(right_view, points_right)
# print("Number of brightest point in left image", len(points_left))
# print("Number of brightest points in epiline",len(brightest_points))



# cv2.imshow("left_picture", left_view)
# cv2.imshow("right_picture", right_view)

# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows