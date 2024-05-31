import os
import cv2
import numpy as np
import math
import time


def slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def find_center_corners(squares):
    dists = {}
    for pts in range(3):
        for pts1 in range(3):
            pt = squares[pts]
            pt1 = squares[pts1]
            dist = np.sqrt(((pt[0] - pt1[0]) ** 2) + ((pt[1] - pt1[1]) ** 2))
            dists[dist] = [pt, pt1]

    corners = dists[max(dists)]

    for ct in squares:
        if ct not in corners:
            center = ct
    return center, corners


def filter_cnt_area(lower_limit, higher_limit, cnt_sorted):
    squares = []
    for cnt in cnt_sorted:
        area = cv2.contourArea(cnt)
        if lower_limit < area < higher_limit:
            approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                xc = x + (w / 2)
                yc = y + (h / 2)
                squares.append((xc, yc))
    return squares


def find_angle_with_horizontal_axis(center, corner):
    m1 = slope(center, corner)
    m2 = 0
    Calc_angle = (m2 - m1) / (1 + m1 * m2)
    rad_angle = math.atan(Calc_angle)
    degree_angle = round(math.degrees(rad_angle))
    return degree_angle


def rotate_image(array, angle):
    height, width = array.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))
    return rotated_mat


if __name__ == '__main__':
    folder_path = r"C:\Users\kalyani chagala\Downloads\may4_qr_images\may4_qr_images"
    files = os.listdir(folder_path)

    out_dir = r"C:\Users\kalyani chagala\Downloads\may4_qr_images\may4_qr_images_out"
    os.makedirs(out_dir, exist_ok=True)

    for file in files:
        start = time.time()
        img = cv2.imread(os.path.join(folder_path, file))
        img_resize = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        img_rotate = img.copy()

        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        not_img = cv2.bitwise_not(img_gray)
        ret, thresh_not = cv2.threshold(not_img, 120, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        img_erosion = cv2.erode(thresh_not, kernel, iterations=1)

        contours_er, hierarchy_er = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_sorted = sorted(contours_er, key=cv2.contourArea, reverse=True)

        squares = filter_cnt_area(3500, 5000, cnt_sorted)
        if len(squares) != 3:
            squares = filter_cnt_area(1500, 3000, cnt_sorted)

        if len(squares) == 3:  ### squares found,continue rotating the qr
            center, corners = find_center_corners(squares)
            degree_angle = find_angle_with_horizontal_axis(center, corners[0])

            im_resize = cv2.circle(img_resize, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
            im_resize = cv2.circle(img_resize, (int(corners[0][0]), int(corners[0][1])), 3, (0, 0, 255), -1)
            im_resize = cv2.circle(img_resize, (int(corners[1][0]), int(corners[1][1])), 3, (0, 0, 255), -1)

            if degree_angle < 0:
                angle_adjusted = 180 + degree_angle
                rotated = 360 - angle_adjusted

                img_resize = rotate_image(img_resize, rotated)
                img_rotate = rotate_image(img_rotate, rotated)

            if degree_angle > 0:
                rotated = 360 - degree_angle

                img_resize = rotate_image(img_resize, rotated)
                img_rotate = rotate_image(img_rotate, rotated)

            img_blank = np.zeros(img_resize.shape, np.uint8)
            blue_mask = np.all(img_resize == [0, 0, 255], axis=-1)
            img_blank[blue_mask] = [255, 255, 255]
            img_gray = cv2.cvtColor(img_blank, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
            contours_dots, hierarchy_dots = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            ### Find the orientation of blue dots
            bboxes = []
            for ct in contours_dots:
                bbox = cv2.boundingRect(ct)
                bboxes.append((bbox[0], bbox[1]))

            oriented_center, oriented_corners = find_center_corners(bboxes)

            center_qr = oriented_center
            corner_qr_pt1 = oriented_corners[0]
            corner_qr_pt2 = oriented_corners[1]

            ######case1:
            if ((corner_qr_pt1[0] > center_qr[0]) and (abs(corner_qr_pt1[1] - center_qr[1]) < 50) and (
                    corner_qr_pt2[1] > center_qr[1]) and (abs(corner_qr_pt2[0] - center_qr[0]) < 50)) or (
                    (corner_qr_pt2[0] > center_qr[0]) and (abs(corner_qr_pt2[1] - center_qr[1]) < 50) and (
                    corner_qr_pt1[1] > center_qr[1]) and (abs(corner_qr_pt1[0] - center_qr[0]) < 50)):
                qr_type = "type1"

            #####case2:
            if ((center_qr[0] > corner_qr_pt1[0]) and (abs(center_qr[1] - corner_qr_pt1[1]) < 50) and (
                    corner_qr_pt2[1] > center_qr[1]) and (abs(corner_qr_pt2[0] - center_qr[0]) < 50)) or (
                    (center_qr[0] > corner_qr_pt2[0]) and (abs(center_qr[1] - corner_qr_pt2[1]) < 50) and (
                    corner_qr_pt1[1] > center_qr[1]) and (abs(corner_qr_pt1[0] - center_qr[0]) < 50)):
                qr_type = "type2"

            #####case3:
            if ((corner_qr_pt1[0] > center_qr[0]) and (abs(center_qr[1] - corner_qr_pt1[1]) < 50) and (
                    center_qr[1] > corner_qr_pt2[1]) and (abs(corner_qr_pt2[0] - center_qr[0]) < 50)) or (
                    (corner_qr_pt2[0] > center_qr[0]) and (abs(center_qr[1] - corner_qr_pt2[1]) < 50) and (
                    center_qr[1] > corner_qr_pt1[1]) and (abs(corner_qr_pt1[0] - center_qr[0]) < 50)):
                qr_type = "type3"

            ####case4:
            if ((center_qr[0] > corner_qr_pt1[0]) and (abs(center_qr[1] - corner_qr_pt1[1]) < 50) and (
                    center_qr[1] > corner_qr_pt2[1]) and (abs(corner_qr_pt2[0] - center_qr[0]) < 50)) or (
                    (center_qr[0] > corner_qr_pt2[0]) and (abs(center_qr[1] - corner_qr_pt2[1]) < 50) and (
                    center_qr[1] > corner_qr_pt1[1]) and (abs(corner_qr_pt1[0] - center_qr[0]) < 50)):
                qr_type = "type4"

            if qr_type == "type1":
                final_rotate_img = img_rotate
            elif qr_type == "type2":
                final_rotate_img = rotate_image(img_rotate, 90)
            elif qr_type == "type3":
                final_rotate_img = rotate_image(img_rotate, -90)
            elif qr_type == "type4":
                final_rotate_img = rotate_image(img_rotate, -180)

            end = time.time()
            # print(end - start)

            cv2.imwrite(os.path.join(out_dir, file), final_rotate_img)
        else:
            print("squares not found")
