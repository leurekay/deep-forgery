from generate import *
color_list=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0)]

# img_dir=r"C:\Users\ye.liu01\Documents\data\deep-forgery\data\raw_data"
img_dir=r"C:\Users\ye.liu01\Documents\data\deep-forgery\data\ps"
output_dir=r"C:\Users\ye.liu01\Documents\data\deep-forgery\data\output"
label_dir=r"C:\Users\ye.liu01\Documents\data\deep-forgery\data\label"
filename_list=os.listdir(img_dir)[100:110]
# path_list=[os.path.join(img_dir,x) for x in filename_list]
# 创建 SIFT 特征检测器
sift = cv2.SIFT_create()
# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def extract_region_keypoints(keypoints, descriptors, pts):
    region_keypoints = []
    region_descriptors = []

    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
            region_keypoints.append(keypoint)
            region_descriptors.append(descriptors[i])
    return region_keypoints, np.array(region_descriptors)


def extract_region_keypoints_simple(keypoints, descriptors, pts):
    region_keypoints = []
    region_descriptors = []
    x1,y1,x2,y2,x3,y3,x4,y4=pts.flatten()
    left=max(x1,x4)+5
    top=max(y1,y2)+5
    right=min(x2,x3)-5
    bottom=min(y3,y4)-5

    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        if x>left and x<right and y>top and y<bottom:
            region_keypoints.append(keypoint)
            region_descriptors.append(descriptors[i])
    return region_keypoints, np.array(region_descriptors)


def compare_keypoints(descriptors1, descriptors2):

    
    # 匹配描述符
    matches = bf.match(descriptors1, descriptors2)
    # 计算匹配得分（距离越小，匹配越好）
    score = sum([match.distance for match in matches]) / len(matches) if matches else float('inf')
    return score, matches

def two_patch_sift_sim(keypoints, descriptors,pts1,pts2):
    # 提取两个区域内的特征点和描述符
    keypoints1, descriptors1 = extract_region_keypoints(keypoints, descriptors, pts1)
    keypoints2, descriptors2 = extract_region_keypoints(keypoints, descriptors, pts2)
    # 比较两个区域内的特征点
    if abs(len(descriptors2)-len(descriptors1))>5:
        return 999999
    score, matches = compare_keypoints(descriptors1, descriptors2)
    return score

def patch_sift_sim(keypoints, descriptors,patchs):
    box=[]
    count=0
    n_rows=patchs.shape[0]
    n_columns=patchs.shape[1]
    for i in range(n_rows):
        for j in range(n_columns):
            ij_index=i*n_rows+j
            for m in range(n_rows):
                for n in range(n_columns):
                    mn_index=m*n_rows+n
                    if mn_index<=ij_index:
                        continue

                    count+=1
                    if count%1000==0:
                        print(count)
                    src_points=patchs[i,j,:8].reshape([4,2])
                    dst_points=patchs[m,n,:8].reshape([4,2])
                    score=two_patch_sift_sim(keypoints,descriptors,src_points,dst_points)
                    box.append([score,(i,j),(m,n)])
    return box

for filename in filename_list:
    img_path=os.path.join(img_dir,filename)
    print(img_path)
    image=read_img(img_path)
    mask=clean(image)

    crosspoint_matrix,line_list_h,line_list_v=get_element(mask)
    crosspoint_list=crosspoint_matrix.reshape((-1,2))
    crosspoint_list=[tuple(x) for x in crosspoint_list.astype(np.int64).tolist()]
    patchs=get_patchs(crosspoint_matrix).astype(np.float32)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    print("keypoints number : ",len(keypoints))
    sim_score_list=patch_sift_sim(keypoints,descriptors,patchs)
    sim_score_list=sorted(sim_score_list,key=lambda x : x[0])
    print(sim_score_list[:4])

    output_image=copy.deepcopy(image)
    for i,(score,src_idx,dst_idx) in enumerate(sim_score_list[:5]):
        src_points=patchs[src_idx[0],src_idx[1],:8].reshape((4,2)).astype(np.int32)
        dst_points=patchs[dst_idx[0],dst_idx[1],:8].reshape((4,2)).astype(np.int32)
        color = color_list[i]

        # 绘制四边形
        cv2.polylines(output_image, [src_points], isClosed=True, color=color, thickness=10)
        cv2.polylines(output_image, [dst_points], isClosed=True, color=color, thickness=10)      

    label_img=read_img(os.path.join(label_dir,filename))
    caoncat_img=np.concatenate([output_image,label_img],axis=0)
    out_path=os.path.join(output_dir,filename)
    cv2.imwrite(out_path, caoncat_img)
        

