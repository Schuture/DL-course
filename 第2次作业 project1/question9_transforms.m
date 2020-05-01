% 载入原数据（0-255的灰度图），进行数据增强后存储起来
tic;
load digits.mat
X = transforms(X);
y = repmat(y, 5, 1);
save augmented_digits.mat X Xtest Xvalid y ytest yvalid
toc;

function [X] = transforms(X)

[nsamples, nfeatures] = size(X);
nrow = round(sqrt(nfeatures));
ncol = round(sqrt(nfeatures));

for i=1:nsamples
    image = X(i,:);
    image = reshape(image, nrow, ncol);
    
    image_pos_15degree = imrotate(image, 15, 'bilinear'); % 逆时针旋转15度
    image_neg_15degree = imrotate(image, -15, 'bilinear'); % 顺时针旋转15度
    image_pos_15degree = imresize(image_pos_15degree, [nrow ncol], 'bilinear'); % 缩放旋转后的图像
    image_neg_15degree = imresize(image_neg_15degree, [nrow ncol], 'bilinear'); % 缩放旋转后的图像
    image_left_trans = [image(:,3:ncol) zeros(nrow, 2)]; % 左移2个像素后的图像
    image_right_trans = [zeros(nrow, 2) image(:,1:ncol-2)]; % 右移2个像素后的图像
    
    % 将图像拉直为向量，便于存储
    image_pos_15degree = reshape(image_pos_15degree, 1, nfeatures);
    image_neg_15degree = reshape(image_neg_15degree, 1, nfeatures);
    image_left_trans = reshape(image_left_trans, 1, nfeatures);
    image_right_trans = reshape(image_right_trans, 1, nfeatures);
    
    % 存储变换后的图像
    X(nsamples+i,:) = image_pos_15degree;
    X(2*nsamples+i,:) = image_neg_15degree;
    X(3*nsamples+i,:) = image_left_trans;
    X(4*nsamples+i,:) = image_right_trans;
end
end