from typing import Any
import cv2
import glob
import os
import numpy as np

def read_img(img_path):
    img = cv2.imread(img_path)
    return img

def detect_corner(img):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测角点
    dst = cv2.cornerHarris(gray, blockSize=15, ksize=3, k=0.04)

    return dst


def detect_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建SIFT检测器
    sift = cv2.SIFT_create()

    # 检测特征点并计算描述子
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(keypoints[0])
    print(len(keypoints))
    print(descriptors.shape)
    print(len(descriptors))
    cross_point_index_list=[]
    cross_point_list=[]
    for i,keypoint in enumerate(keypoints):
        if keypoint.size>20:
            cross_point_list.append(keypoint)
            cross_point_index_list.append(i)
    print(len(cross_point_list))
    # 在图像上绘制特征点
    image_with_keypoints = cv2.drawKeypoints(img, cross_point_list, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
    return image_with_keypoints 


def find_contour(image):
    # 取反操作，确保0像素变为255，其他像素变为0
    inverted_image = cv2.bitwise_not(image)

    # 查找所有连通区域
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于显示结果
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 绘制所有连通区域
    for contour in contours:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
    return contours

    # # 显示结果
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Inverted Image', inverted_image)
    # cv2.imshow('Connected Components', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def remove_small_area(image,threshold_area):
    contours=find_contour(image)
    # 创建一个新的图像用于显示结果
    output_image = image.copy()

    # 遍历每个轮廓
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < threshold_area:
            # 将轮廓内的所有点赋值为100
            cv2.drawContours(output_image, [contour], -1, 255, thickness=cv2.FILLED)   
    return output_image



def clean(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 定义结构元素
    # 自适应阈值处理
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # mask = 255 - mask

    #第一次去除横线竖线以外的噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #通过计算联通区域，去除面积小的
    mask=remove_small_area(mask,threshold_area=200)

    kernel2 = np.ones((6, 2), np.uint8)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)


    # 膨胀操作
    # mask = cv2.dilate(mask, kernel2, iterations=1)

    # 腐蚀操作
    # mask = cv2.erode(mask, kernel2, iterations=1)


    return mask

class Generator:
    def __init__(self,img_dir) -> None:
        filename_list=os.listdir(img_dir)[100:]
        self.path_list=[os.path.join(img_dir,x) for x in filename_list]
        self.img_list=[read_img(x) for x in self.path_list]




if __name__=="__main__":
    img_dir=r"C:\Users\ye.liu01\Documents\data\deep-forgery\data\raw_data"
    generator=Generator(img_dir)
    print(generator.path_list)
    img_list=generator.img_list
    img=img_list[1][:7000,:7000,:]
    print(img.shape)
    # corner=detect_corner(img)
    # print(corner)
    # # 膨胀角点
    # corner = cv2.dilate(corner, None,iterations=2)
    # print(sum(sum(corner)))
    # # 标记角点
    # img[corner > 0.01 * corner.max()] = [0, 0, 255]  # 将角点标记为红色

    image_with_keypoints=detect_feature(img)



    # 显示带有角点标记的图片
    # cv2.imshow('Corner detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('sift_output_image.jpg', image_with_keypoints)
