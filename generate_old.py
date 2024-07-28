from typing import Any
import cv2
import glob
import os
import numpy as np
import copy

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

    # 在图像上绘制特征点
    image_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
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

    kernel2 = np.ones((3, 3), np.uint8)
    # 膨胀操作
    mask = cv2.dilate(mask, kernel2, iterations=1)
    #腐蚀操作
    # mask = cv2.erode(mask, kernel2, iterations=5)


    # kernel3= np.ones((1, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel3, iterations=3)
    return mask

def split_array(nums):
    threshold=40
    box=[]
    s=-1
    e=-1
    for i,v in enumerate(nums):
        if v>threshold:
            if s==-1:
                s=i
            else:
                e=i
        else:
            if s!=-1 and e!=-1:
                box.append([s,e])
                s=-1
                e=-1
    return box


def fitting_line(image,mask):
    image=image*mask
    y_indices, x_indices = np.where(image > 0)
    points = np.array(list(zip(x_indices, y_indices)))

    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((image.shape[1] - x) * vy / vx) + y)
    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.line(output_image, (image.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)
    return lefty,righty
    

def get_line(mask):
    kernel_x= np.ones((1, 5), np.uint8) #水平直线
    mask_x= cv2.dilate(mask, kernel_x, iterations=3)
    kernel_y= np.ones((5, 1), np.uint8) #竖直直线
    mask_y = cv2.dilate(mask, kernel_y, iterations=3)

    #获取直线在y轴上的大致投影区间
    y_dist=np.mean(255-mask_x,axis=1)
    y_intervals=split_array(y_dist)
    print(y_intervals)

    #X轴上
    x_dist=np.mean(255-mask_y,axis=0)
    x_intervals=split_array(x_dist)
    print(x_intervals)


    #计算得到水平直线
    # render_image=copy.deepcopy(mask)
    line_box=[]
    for interval in y_intervals:
        valid_mask=np.zeros_like(mask)
        valid_mask[interval[0]:interval[1]+1,:]=1
        lefty,righty=fitting_line(mask,valid_mask)
        line_box.append([lefty,righty])


    return line_box



def merge_lines(lines, max_gap=10):
    """
    合并相邻的直线段。
    
    参数:
    - lines: 检测到的直线段。
    - max_gap: 合并直线段的最大间隔。
    
    返回:
    - merged_lines: 合并后的直线段。
    """
    if len(lines) == 0:
        return []
    
    # 将线段转换为二维数组
    lines = np.array(lines)
    
    merged_lines = []
    current_line = lines[0]
    
    for i in range(1, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        cx1, cy1, cx2, cy2 = current_line[0]
        
        # 计算线段之间的距离
        dist1 = np.sqrt((x1 - cx2) ** 2 + (y1 - cy2) ** 2)
        dist2 = np.sqrt((x2 - cx1) ** 2 + (y2 - cy1) ** 2)
        
        if dist1 < max_gap or dist2 < max_gap:
            # 合并线段
            current_line = np.array([[min(cx1, x1), min(cy1, y1), max(cx2, x2), max(cy2, y2)]])
        else:
            merged_lines.append(current_line)
            current_line = lines[i]
    merged_lines.append(current_line)
    return merged_lines

def line_detect(image):
    # 创建 LSD 检测器
    lsd = cv2.createLineSegmentDetector(0)

    # 检测直线段
    lines,_,_,_ = lsd.detect(image)
    for x in lines:
        print(x)
    print("len lines ",len(lines))

    # 合并直线段
    merged_lines = merge_lines(lines, max_gap=30)
    # 将合并后的直线段绘制到图像上
    output_image = np.copy(image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像
    for line in merged_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return output_image




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
    img=img_list[3][:7000,:7000,:]
    print(img.shape)
    # corner=detect_corner(img)
    # print(corner)
    # # 膨胀角点
    # corner = cv2.dilate(corner, None,iterations=2)
    # print(sum(sum(corner)))
    # # 标记角点
    # img[corner > 0.01 * corner.max()] = [0, 0, 255]  # 将角点标记为红色

    image_with_keypoints=detect_feature(img)
    mask=clean(img)
    line_list=get_line(mask)

    output_image=copy.deepcopy(img)
    for lefty,righty in line_list:
        cv2.line(output_image, (img.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)

    # line_image=line_detect(mask)



    # 显示带有角点标记的图片
    # cv2.imshow('Corner detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('output_image.jpg', output_image)



