function [f,g1,g2] = question10_ConvNet(w1,w2,X,y,kernel_size,nHidden,nLabels)

X = X(:,2:257); % ��������һ��ƫ����
[nInstances,nVars] = size(X); % ������������������
height = round(sqrt(nVars));
width = nVars / height;

% Form Weights
nkernels = nHidden(1);
inputWeights = reshape(w1(1:kernel_size^2*nkernels), nkernels, kernel_size^2); % ��������
nHidden(1) = nVars * nkernels; % ����ͼչ����ĳ���
offset = 0; % һ��ָ�룬������һ��Ĳ��������￪ʼȡ
for h = 2:length(nHidden)
  % ��Ȩ������w��һ���ֽ�ȡ������ת��Ϊ��һ��ľ�����ʽ
  hiddenWeights{h-1} = reshape(w2(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); % ÿ��ȡһ��Ĳ���������ָ��
end
outputWeights = w2(offset+1:offset+nHidden(end)*nLabels); % ���һ��Ȩ�أ�����
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % ���һ��Ȩ�أ�����

f = 0; % ��ʼ����ʧ����ֵ
if nargout > 1 % ��ʼ���ݶ�
    gInput = zeros(size(w1)); % ��ʼ���������ݶ�
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); % ��ʼ�����ز���ݶ�
    end
    gOutput = zeros(size(outputWeights)); % ��ʼ���������ݶ�
end

% Compute Output
for i = 1:nInstances % ����ÿһ������
    image = reshape(X(i,:), height, width);
    image = padarray(image, [floor(kernel_size/2), floor(kernel_size/2)]);
    for j = 1:nkernels % ���������nkernels������˶�Ӧnkernels������ͼ
        kernel{j} = reshape(inputWeights(j,:), kernel_size, kernel_size);
        featmap{j} = conv2(image, kernel{j}, 'valid');
        ip{1}((j-1)*nVars+1:j*nVars) = reshape(featmap{j}, 1, nVars); % ������ͼ��ֱ
    end
    fp{1} = tanh(ip{1}); % ���������
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1}; % ��һ�������ز�
        fp{h} = tanh(ip{h}); % ���������
    end
    yhat = fp{end} * outputWeights; % �������һ��Ȩ��
    
    relativeErr = yhat - y(i,:); % �в�
    f = f + sum(relativeErr.^2); % ���в�ƽ���ͼӵ�����ʧ��
    
    if nargout > 1
        err = 2 * relativeErr; % �������򴫲�

        % Output Weights
        gOutput = gOutput + (err' * fp{end})'; 

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            % clear backprop % ȥ����һ���Ժ��ٶ�X2.5
            backprop = err' .* (sech(ip{end}) .^ 2 .* outputWeights');
            gHidden{end} = gHidden{end} + repmat(fp{end-1}', 1, nLabels) * backprop;
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end

            % Input Weights����ʵ������һ������Ҫ���о�����ķ��򴫲�����Ϊֻ��һ����
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2; % 1 * 768
            for j = 1:nkernels
                delta{j} = reshape(backprop((j-1)*nVars+1:j*nVars), height, width);
                grad = conv2(image, imrotate(delta{j}, 180, 'bilinear'), 'valid');
                gInput((j-1)*kernel_size^2+1:j*kernel_size^2) = reshape(grad, kernel_size^2, 1);
            end
        else
            % Input Weights
            backprop = err * (sech(ip{end}) .^ 2 .* outputWeights');
            for j = 1:nkernels
                delta{j} = reshape(backprop((j-1)*nVars+1:j*nVars), height, width);
                grad = conv2(image, imrotate(delta{j}, 180, 'bilinear'), 'valid');
                gInput((j-1)*kernel_size^2+1:j*kernel_size^2) = reshape(grad, kernel_size^2, 1);
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1 % ����������ز�֮��Ĳ�������һ��һ�㽫��Ӧ�ݶ�ֵ�Ž�g��
    g1 = gInput; % ������ݶ�
    g2 = zeros(size(w2));
    offset = 0; % һ��ָ�룬������һ����ݶȷŵ�����
    for h = 2:length(nHidden)
        g2(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    g2(offset+1:offset+nHidden(end)*nLabels) = gOutput(:); % ������ݶ�
end
