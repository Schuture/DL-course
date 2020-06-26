''' 根据gt文件夹中的标注信息，将图像的文字框部分提取出来并保存为一个数据集 '''

import os
import random
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    
    if out_img.shape[0] > 1.5 * out_img.shape[1]: # 高大于宽的1.5倍，可以认为是一列文字
        trans_img = cv2.transpose(out_img)
        out_img = cv2.flip(trans_img, 0) # 逆时针旋转90度

    return out_img


if __name__ == '__main__':
    mode = 'train' # 或者'test'
    img_dataset_path = '/root/CYX_Space/data/' + mode + '/img' # 数据集图片目录
    img_paths = os.listdir(img_dataset_path)
    gt_dataset_path = '/root/CYX_Space/data/' + mode + '/gt' # 数据集的gt目录
    output_img_path = '/root/CYX_Space/data/' + mode + '_text_img' # 经过提取后，截取的文字部分的图片放在这个目录
    output_text_path = '/root/CYX_Space/CRNN/lib/dataset/txt/' # 将每张图片的标注保存起来
    output_text_file = mode + '_own.txt'
    
    if os.path.exists(output_img_path): 
        shutil.rmtree(output_img_path) # 先清理掉保存图片的文件夹，以免有其他文件
    os.mkdir(output_img_path) # 重新创建保存图片的文件夹
    with open(os.path.join(output_text_path, output_text_file), 'w', encoding='utf-8') as f:
                f.write('') # 将标注文件其中的内容清空
    
    text_num = 0 # 对剪切出的图像计数
    for iteration, filename in enumerate(img_paths):
        print('\n开始处理第{}张原图。{}\n'.format(iteration+1, filename))
        img_name = filename
        gt_name = filename.split('.')[0] + '.txt'
        img_path = os.path.join(img_dataset_path, img_name)
        gt_path = os.path.join(gt_dataset_path, gt_name)
        
        img = plt.imread(img_path) # 读取图像

        gt_coordinates = []
        gt_labels = []
        with open(gt_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                coordinates = line.split(',')[:8] # 左上，左下，右下，右上
                coordinates = [int(coor) for coor in coordinates]
                gt_coordinates.append(coordinates)
                label = line.split(',')[9][:-1] # 要去掉最后的'\n'
                gt_labels.append(label)
        
        for i in range(len(gt_coordinates)):
            out_img = perspective_affine(img, gt_coordinates[i])
            out_label = gt_labels[i]
            if out_label == '###': # 不关注的区域我们只取十分之一来训练，一方面平衡样本，一方面加速训练
                if random.random() > 0.1:
                    print('忽略一张背景')
                    continue
                else:
                    print('取一张背景')
            img_name = str(text_num) + '.jpg' # png无损保存太大，使用jpg
            cv2.imwrite(os.path.join(output_img_path, img_name), out_img) # cv2.imwrite路径不能有中文
            with open(os.path.join(output_text_path, output_text_file), 'a', encoding='utf-8') as f:
                f.write(img_name + ' ' + out_label + '\n') # 图像名 内容
            text_num += 1
            print('第{}张文本图像保存完毕！'.format(text_num))
        #if iteration == 50:
        #    break














