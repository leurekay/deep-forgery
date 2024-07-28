from typing import Any
import cv2
import glob
import os
import numpy as np
import copy
import random
import json

StandRatio=2

def read_img(img_path):
    img = cv2.imread(img_path)
    return img



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
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个彩色图像用于显示结果
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 绘制所有连通区域
    for contour in contours:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
    return contours

def remove_small_area(image,threshold_area):
    contours=find_contour(image)
    # 创建一个新的图像用于显示结果
    output_image = image.copy()
    # 遍历每个轮廓
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < threshold_area:
            cv2.drawContours(output_image, [contour], -1, 255, thickness=cv2.FILLED)   
    return output_image

def gradient(image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Sobel 算子计算 X 方向梯度
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # 使用 Sobel 算子计算 Y 方向梯度
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 将梯度图转换为 uint8 类型
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)


    # _, sobel_x = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
    sobel_x=cv2.adaptiveThreshold(
        sobel_x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)
    sobel_y=cv2.adaptiveThreshold(
        sobel_y, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)
    sobel_x=255-sobel_x
    sobel_y=255-sobel_y

    sobel_x=remove_small_area(sobel_x,300)
    sobel_y=remove_small_area(sobel_y,300)
    
    sobel_x=cv2.dilate(sobel_x, np.ones((3, 3), np.uint8), iterations=1)
    sobel_y=cv2.dilate(sobel_y, np.ones((3, 3), np.uint8), iterations=1)

    sobel_x = cv2.morphologyEx(sobel_x, cv2.MORPH_CLOSE, np.ones((3, 3)))
    sobel_y = cv2.morphologyEx(sobel_y, cv2.MORPH_CLOSE, np.ones((3, 3)))
    
    # sobel_y = cv2.morphologyEx(sobel_y, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    sobel_x=remove_small_area(sobel_x,200)
    sobel_y=remove_small_area(sobel_y,200)
    return sobel_x,sobel_y

def clean(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    gray = cv2.equalizeHist(gray)
    # 定义结构元素
    # 自适应阈值处理
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # mask = 255 - mask
    #第一次去除横线竖线以外的噪声
    kernel = np.ones((5, 5), np.uint8)
    # 应用开运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #通过计算联通区域，去除面积小的
    
    mask=remove_small_area(mask,threshold_area=500)

    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 15), np.uint8))

    # mask = cv2.erode(mask, np.ones((1, 2), np.uint8), iterations=2)

    # 膨胀操作
    # mask = cv2.dilate(mask, np.ones((3, 1), np.uint8), iterations=1)
    #腐蚀操作
    # mask = cv2.erode(mask, kernel2, iterations=5)
    # kernel3= np.ones((1, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel3, iterations=3)
    mask=remove_small_area(mask,threshold_area=300)

    return mask

def split_array(nums,threshold):
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
    

def get_line(mask,threshold):
    #从边框的mask图片中，拟合出每条边框的直线
    # kernel_y= np.ones((5, 1), np.uint8) #竖直直线
    # mask_y = cv2.dilate(mask, kernel_y, iterations=3)
    # #X轴上
    # x_dist=np.mean(255-mask_y,axis=0)
    # x_intervals=split_array(x_dist)
    # print(x_intervals)

    kernel_x= np.ones((1, 5), np.uint8) #水平直线
    mask_x= cv2.dilate(mask, kernel_x, iterations=3)
    #获取直线在y轴上的大致投影区间
    y_dist=np.mean(255-mask_x,axis=1)
    y_intervals=split_array(y_dist,threshold)

    #计算得到水平直线[x1,y1,x2,y2]
    line_box=[]
    for interval in y_intervals:
        valid_mask=np.zeros_like(mask)
        valid_mask[interval[0]:interval[1]+1,:]=1
        lefty,righty=fitting_line(mask,valid_mask)
        line_box.append([0, lefty,mask.shape[1] - 1, righty])
    return line_box


def two_line_intersection(line_a,line_b):
    x1, y1, x2, y2=line_a
    x3, y3, x4, y4=line_b
    # 计算直线1和直线2的斜率
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    # 如果两条直线平行，则没有交点
    if m1 == m2:
        return None
    # 计算直线1的截距
    b1 = y1 - m1 * x1 if m1 != float('inf') else None
    # 计算直线2的截距
    b2 = y3 - m2 * x3 if m2 != float('inf') else None
    # 计算交点
    if m1 == float('inf'):
        # 直线1垂直于x轴
        x = x1
        y = m2 * x + b2 if b2 is not None else y3
    elif m2 == float('inf'):
        # 直线2垂直于x轴
        x = x3
        y = m1 * x + b1 if b1 is not None else y1
    else:
        # 两条直线都不垂直于x轴
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    return (int(x), int(y))

def get_element(mask):
    line_list_h=get_line(mask,threshold=100) #所有水平直线
    line_list_v=get_line(mask.T,threshold=100)
    line_list_v=[[x[1],x[0],x[3],x[2]] for x in line_list_v]# 所有竖直直线

    crosspoint_list=[]
    crosspoint_matrix = np.zeros((len(line_list_h), len(line_list_v), 2))
    for i,line_h in enumerate(line_list_h):
        for j,line_v in enumerate(line_list_v):
            crosspoint=two_line_intersection(line_h,line_v)
            crosspoint_list.append(crosspoint)
            crosspoint_matrix[i,j,:]=list(crosspoint)
    return crosspoint_matrix,line_list_h,line_list_v


def get_patchs(crosspoint_matrix):
    left_upper_mat=crosspoint_matrix[:-1,:-1]
    right_upper_mat=crosspoint_matrix[:-1:,1:]
    right_bottom_mat=crosspoint_matrix[1:,1:]
    left_bottom_mat=crosspoint_matrix[1:,:-1]

    size_mat=right_bottom_mat-left_upper_mat
    print("average width : ",np.mean(size_mat[:,:,0]))
    print("average height : ",np.mean(size_mat[:,:,1]))
    area_mat=size_mat[:,:,0]*size_mat[:,:,1]
    ratio_mat=size_mat[:,:,1]/size_mat[:,:,0]

    valid_mask = np.where(abs(ratio_mat-StandRatio)<0.2 , 1, 0)
    concat_mat=np.concatenate([left_upper_mat,right_upper_mat,right_bottom_mat,left_bottom_mat,np.expand_dims(valid_mask, axis=-1)],axis=-1)
    return concat_mat


def copy_move_same_image(image, src_pts, dst_pts):
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    # 变换图像块
    transformed_region = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    # 创建一个掩膜，表示变换后的位置
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(int), (255, 255, 255))
    # 将变换后的图像块粘贴到目标位置
    result = cv2.bitwise_and(transformed_region, mask) + cv2.bitwise_and(image, cv2.bitwise_not(mask))
    return result,mask[:,:,0]

def copy_move(src_image, dst_image, src_pts, dst_pts):
    pass


def apply_blur_to_polygon(image, points, blur_strength=(21, 21)):
    """
    对图像中指定的四边形框区域进行模糊处理,掩盖PS痕迹。

    参数:
    image (numpy.ndarray): 输入图像。
    points (numpy.ndarray): 四边形框的顶点坐标，形状为 (4, 2)。
    blur_strength (tuple): 高斯模糊核的大小，默认为 (21, 21)。

    返回:
    numpy.ndarray: 处理后的图像。
    """
    # 创建一个与图像大小相同的黑色掩码图像
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 在掩码图像上绘制一个白色填充四边形
    cv2.drawContours(mask, [points],-1,255,20)
    cv2.imwrite("aaaaaaaaaaa.jpg",mask)

    # 对掩码图像进行高斯模糊
    blurred_mask = cv2.GaussianBlur(mask, blur_strength, 0)

    # 将模糊的掩码应用于原始图像
    feathered_image = image.copy()
    for i in range(3):  # 对每个通道应用掩码
        feathered_image[:, :, i] = cv2.addWeighted(image[:, :, i], 0.5, blurred_mask, 0.5, 0)

    return feathered_image

class Generator:
    def __init__(self,img_dir,output_dir) -> None:
        filename_list=os.listdir(img_dir)[0:]
        self.path_list=[os.path.join(img_dir,x) for x in filename_list]
        # self.img_list=[read_img(x) for x in self.path_list]
        self.output_dir=output_dir
        
        self.img_patchs_pair_list=[]
        for i,filename in enumerate(filename_list):
            print(i)
            img_path=os.path.join(img_dir,filename)
            img,patchs,mask,line_point_img=self.process(img_path)
            self.img_patchs_pair_list.append((filename,img,patchs))
    
    def ps_image(self,src_img_idx,dst_img_idx):
        dst_name,dst_img,dst_patchs=self.img_patchs_pair_list[dst_img_idx]
        src_name,src_img,src_patchs=self.img_patchs_pair_list[src_img_idx]
        src_valid_coordinates = np.argwhere(src_patchs[:,:,8] == 1).tolist()
        dst_valid_coordinates = np.argwhere(dst_patchs[:,:,8] == 1).tolist()
        src_pick_coordinate=random.choice(src_valid_coordinates)
        if src_img_idx==dst_img_idx:
            src_valid_coordinates.remove(src_pick_coordinate)
            dst_pick_coordinate=random.choice(src_valid_coordinates)
        else:
            dst_pick_coordinate=random.choice(dst_valid_coordinates)
        src_points=src_patchs[src_pick_coordinate[0],src_pick_coordinate[1],:8].reshape([4,2])
        dst_points=dst_patchs[dst_pick_coordinate[0],dst_pick_coordinate[1],:8].reshape([4,2])
        fake_image,mask=copy_move_same_image(src_img,src_points,dst_points)
        # fake_image=apply_blur_to_polygon(fake_image,dst_points.astype(np.int32))
        meta_data={"src_name":src_name,
                   "dst_name":dst_name,
                   "src_point":src_points.tolist(),
                   "dst_point":dst_points.tolist()}
        return fake_image,mask,meta_data
        
    def generate_train_data(self,number,output_dir):
        meta_data_list=[]
        N=len(self.path_list)
        for i in range(number):
            idx=10000+i
            img_idx=i%N
            try:
                fake_image,mask,meta_data=self.ps_image(img_idx,img_idx)
            except Exception as e:
                continue
            cv2.imwrite(os.path.join(output_dir,"{}.jpg".format(idx)), fake_image)
            cv2.imwrite(os.path.join(output_dir,"{}_label.jpg".format(idx)), mask)
            meta_data["idx"]=idx
            meta_data_list.append(meta_data)
        with open(os.path.join(output_dir,"trian.json"), 'w', encoding="utf-8") as f:
            json.dump(meta_data_list, f, indent=4, ensure_ascii=False)
            

    def process(self,img_path):
        filename=os.path.basename(img_path)
        img=read_img(img_path)
        print(img.shape)

        # sobel_x,sobel_y=gradient(img)
        # sobel_path=os.path.join(self.output_dir,filename.replace(".jpg","_sobel.jpg"))
        # sobel_x_y=np.concatenate([sobel_x,sobel_y],axis=0)
        # cv2.imwrite(sobel_path, sobel_x_y)

        mask=clean(img)
        mask_path=os.path.join(self.output_dir,filename.replace(".jpg","_mask.jpg"))
        cv2.imwrite(mask_path, mask)
        crosspoint_matrix,line_list_h,line_list_v=get_element(mask)
        crosspoint_list=crosspoint_matrix.reshape((-1,2))
        crosspoint_list=[tuple(x) for x in crosspoint_list.astype(np.int64).tolist()]
        line_point_img=copy.deepcopy(img)
        for x in line_list_h:
            cv2.line(line_point_img, (x[0], x[1]), (x[2], x[3]), (0, 255, 0), 2)
        for x in line_list_v:
            cv2.line(line_point_img, (x[0], x[1]), (x[2], x[3]), (255, 0, 0), 2)
        for crosspoint in crosspoint_list:
            cv2.circle(line_point_img, crosspoint, 5, (0, 0, 255), -1)
        crosspoint_path=os.path.join(self.output_dir,filename.replace(".jpg","_crosspoint.jpg"))
        cv2.imwrite(crosspoint_path, line_point_img)
        patchs=get_patchs(crosspoint_matrix)
        return img,patchs,mask,line_point_img



if __name__=="__main__":
    img_dir=r"data/raw_data"
    output_dir=r"data/generate"
    train_data_dir=r"data/train_data"
    generator=Generator(img_dir,output_dir)
    print(generator.path_list)
    fake_image,mask,meta_data=generator.ps_image(3,3)
    print(meta_data)
    generator.generate_train_data(1000,train_data_dir)

    # img=read_img(generator.path_list[10])
    # print(img.shape)


    # image_with_keypoints=detect_feature(img)
    # mask=clean(img)
    # cv2.imwrite('output_mask.jpg', mask)

    # crosspoint_matrix,line_list_h,line_list_v=get_element(mask)
    # crosspoint_list=crosspoint_matrix.reshape((-1,2))
    # crosspoint_list=[tuple(x) for x in crosspoint_list.astype(np.int64).tolist()]
    # output_image=copy.deepcopy(img)
    # for x in line_list_h:
    #     cv2.line(output_image, (x[0], x[1]), (x[2], x[3]), (0, 255, 0), 2)
    # for x in line_list_v:
    #     cv2.line(output_image, (x[0], x[1]), (x[2], x[3]), (255, 0, 0), 2)
    # for crosspoint in crosspoint_list:
    #     cv2.circle(output_image, crosspoint, 5, (0, 0, 255), -1)
    # cv2.imwrite('output_image.jpg', output_image)


    # patchs=get_patchs(crosspoint_matrix)
    # src_idx=[2,5]
    # dst_idx=[4,7]
    # src_points=patchs[src_idx[0],src_idx[1],:8].reshape([4,2])
    # dst_points=patchs[dst_idx[0],dst_idx[1],:8].reshape([4,2])
    # fake_image=copy_move_same_image(img,src_points,dst_points)
    # cv2.imwrite('fake_image.jpg', fake_image)




    # 显示带有角点标记的图片
    # cv2.imshow('Corner detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    




