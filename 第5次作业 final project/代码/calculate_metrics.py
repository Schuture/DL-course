import os

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint #多边形


def sort_points(points):
    ''' 将四个顶点按照 左上，左下，右上，右下 的顺序排好，假设四边形较扁 '''
    vertex = [None, None, None, None]
    # 朝右的为y坐标轴
    ys = sorted([points[0][1], points[1][1], points[2][1], points[3][1]])
    left, right = [], [] # 两个左边的点以及两个右边的点
    for i in range(4):
        point = points[i]
        if ys.index(point[1]) <= 1 and len(left) < 2: # 较大的两个，后面的条件是防止两个y坐标相同
            left.append(point)
        else:
            right.append(point)
    if left[0][0] < left[1][0]:
        vertex[0], vertex[1] = left[0], left[1]
    else:
        vertex[0], vertex[1] = left[1], left[0]
    if right[0][0] < right[1][0]:
        vertex[2], vertex[3] = right[0], right[1]
    else:
        vertex[2], vertex[3] = right[1], right[0]
    
    return vertex


def perspective_affine(img, bbox):
    '''
    将8个坐标值的四边形图片转化为宽大于高的矩形图片
    Parameters:
        img: ndarray, w * h * 3
        bbox: list, length is 8
    Return:
        out_img: the rectangle image
    '''
    # 源图像中四边形坐标点是逆时针标注的，我们首先找到四个方位的顶点
    points = [[bbox[0],bbox[1]], [bbox[2],bbox[3]],
              [bbox[6],bbox[7]], [bbox[4],bbox[5]]]
    vertex = sort_points(points) # 左上，左下，右上，右下
    h = max(vertex[3][1]-vertex[1][1], vertex[2][1]-vertex[0][1]) # 四边形高
    w = max(vertex[1][0]-vertex[0][0], vertex[3][0]-vertex[2][0]) # 四边形宽
    print('宽',w,'高',h)
    point1 = np.array(vertex, dtype="float32")
    
    #转换后得到矩形的坐标点
    point2 = np.array([[0,0],[w,0],[0,h],[w,h]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(point1, point2)
    out_img = cv2.warpPerspective(img, M, (w,h))
    
    if out_img.shape[0] > out_img.shape[1]: # 高大于宽
        trans_img = cv2.transpose(out_img)
        out_img = cv2.flip(trans_img, 0) # 逆时针旋转90度

    return out_img


def calculate_IoU(coordinate1, coordinate2): 
    '''
    计算任意两个四边形的IoU
    Parameters:
        a, b: list, the coordinate of a region, length is 8
    Return:
        iou: the IoU of the two regions
    '''
    a = np.array(coordinate1).reshape(4, 2)  #四边形二维坐标表示
    # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    poly1 = Polygon(a).convex_hull 
    b = np.array(coordinate2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    
    union_poly = np.concatenate((a,b))  #合并两个box坐标，变为8*2
    #print(union_poly)
    #print(MultiPoint(union_poly).convex_hull)   #包含两四边形最小的多边形点
    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  #相交面积
            #print(inter_area)
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            #print(union_area)
            if union_area == 0:
                iou= 0
            #iou = float(inter_area) / (union_area-inter_area) #错了
            iou=float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积 
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式） 
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    
    return iou


if __name__ == '__main__':
    '''
    第一步：我们先将所有图像输入检测模块，生成检测坐标
    第二步：用这个文件定义的函数求IoU，计算FP, FN样本数
    第三步：将TP样本输入识别模型，这些样本中识别错误的也算入FN中
    '''
    test_img_path = "/root/CYX_Space/data/test/img/" # 测试集图片目录
    img_paths = os.listdir(test_img_path)
    test_gt_path = "/root/CYX_Space/data/test/gt/" # 测试集的gt目录
    # 经过检测网络后，输出的坐标全部以与gt相同的形式放在这个目录
    output_text_path = "/root/CYX_Space/EAST-master/submit/" 
    M = 0 # positive matches
    G = 0 # expected words
    T = 0 # filtered results
    
    #model = Recognizer() # 用于识别检测出来的文本的模型
    for iteration, filename in enumerate(img_paths):
        try:
            img_name = filename
            gt_name = filename.replace('.jpg', '.txt')
            pred_name = 'res_' + gt_name
            img_path = os.path.join(test_img_path, img_name)
            gt_path = os.path.join(test_gt_path, gt_name)
            pred_path = os.path.join(output_text_path, pred_name)

            gt_coordinates = []
            gt_labels = []
            pred_coordinates = []
            with open(gt_path) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    coordinates = line.split(',')[:8] # 左上，左下，右下，右上
                    coordinates = [int(coor) for coor in coordinates]
                    gt_coordinates.append(coordinates)
                    label = line.split(',')[8]
                    gt_labels.append(label)
            with open(pred_path) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    coordinates = line.split(',')[:8] # 左上，左下，右下，右上
                    coordinates = [int(coor) for coor in coordinates]
                    pred_coordinates.append(coordinates)
            G += len(gt_coordinates)
            T += len(pred_coordinates)
            
            corr_pred = []
            for i in range(len(gt_coordinates)):
                for j in range(len(pred_coordinates)):
                    iou = calculate_IoU(gt_coordinates[i], pred_coordinates[j])
                    if iou > 0.5:
                        corr_pred.append((i, j))
                        break # 如果这个预测框被判定为真，则可以不继续判断了
        except Exception as e:
            print('计算图像检测指标出错了。。。')
            print(e)
        
# =============================================================================
#         # 开始进行文字识别
#         correct = 0
#         for i in range(len(corr_pred)):
#             gt_index, pred_index = corr_pred[i] # 当前正确框在gt， pred中的索引
#             gt_text = gt_labels[gt_index]
#             coordinates = pred_coordinates[pred_index]
#             
#             pred_text = model(perspective_affine(img, coordinates)) # 使用矩形图像来文字识别
#             if pred_text == gt_text:
#                 correct += 1
# =============================================================================
        correct = len(corr_pred)
        M += correct
        print('Iteration: {:>5d}/{:>5d}, Correct: {:>3d}, '
              'Total boxes: {:>3d}, Predicted boxes: {:>3d}'.format(iteration, len(img_paths), 
                                                               correct, len(gt_coordinates), 
                                                               len(pred_coordinates)))

    P = M / T
    R = M / G
    F = 2 * P * R / (P + R)
    print('\nTesting finish! Correct: {:>3d}, '
          'Total boxes: {:>3d}, Predicted boxes: {:>3d}'.format(M, G, T))
    print('Precision: {:.2%}, Recall: {:.2%}, F measure: {:.2%}'.format(P, R, F))





























