function [f,gw,gb] = question6_MLP_with_bias(w,b,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X); % ������������������

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)), nVars, nHidden(1)); % ��һ�����
offset = nVars * nHidden(1); % һ��ָ�룬������һ��Ĳ��������￪ʼȡ
for h = 2:length(nHidden)
  % ��Ȩ������w��һ���ֽ�ȡ������ת��Ϊ��һ��ľ�����ʽ
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); % ÿ��ȡһ��Ĳ���������ָ��
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels); % ���һ��Ȩ�أ�����
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % ���һ��Ȩ�أ�����

% Form bias
inputBias = b(1:nHidden(1));
offset = nHidden(1);
for h = 2:length(nHidden)
    hiddenBias{h-1} = b(offset+1:offset+nHidden(h));
    offset = offset + nHidden(h);
end
outputBias = b(offset+1:offset+nLabels);

f = 0; % ��ʼ����ʧ����ֵ
if nargout > 1
    gwInput = zeros(size(inputWeights)); % ��ʼ�������w���ݶ�
    for h = 2:length(nHidden)
       gwHidden{h-1} = zeros(size(hiddenWeights{h-1})); % ��ʼ�����ز�w���ݶ�
    end
    gwOutput = zeros(size(outputWeights)); % ��ʼ�������w���ݶ�
    
    gbInput = zeros(size(inputBias)); % ��ʼ�������b���ݶ�
    for h = 2:length(nHidden)
       gbHidden{h-1} = zeros(size(hiddenBias{h-1})); % ��ʼ�����ز�b���ݶ�
    end
    gbOutput = zeros(size(outputBias)); % ��ʼ�������b���ݶ�
end

% Compute Output
for i = 1:nInstances % ����ÿһ������
    ip{1} = X(i, :) * inputWeights + inputBias'; % ���ݾ�����һ��Ȩ�ز�
    fp{1} = tanh(ip{1}); % ���������
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1} + hiddenBias{h-1}'; % ��һ�������ز�
        fp{h} = tanh(ip{h}); % ���������
    end
    yhat = fp{end} * outputWeights + outputBias'; % �������һ��Ȩ��
    
    relativeErr = yhat - y(i,:); % �в�
    f = f + sum(relativeErr.^2); % ���в�ƽ���ͼӵ�����ʧ��
    
    if nargout > 1
        err = 2 * relativeErr; % �������򴫲�

        % Output Weights
        for c = 1:nLabels
            gwOutput(:,c) = gwOutput(:,c) + err(c) * fp{end}'; % ������ݶȼ���2�����˵����ڶ������
            gbOutput(c) = gbOutput(c) + err(c);
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            for c = 1:nLabels
                % �������򴫲����м��sech˫���������˫�����ҵĵ���
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gwHidden{end} = gwHidden{end} + fp{end-1}' * backprop(c, :);
                gbHidden{end} = gbHidden{end} + backprop(c, :)';
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gwHidden{h} = gwHidden{h} + fp{h}' * backprop;
                gbHidden{h} = gbHidden{h} + backprop';
            end

            % Input Weights
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2;
            gwInput = gwInput + X(i,:)' * backprop;
            gbInput = gbInput + backprop';
        else
            % Input Weights
            for c = 1:nLabels
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gwInput = gwInput + X(i,:)' * backprop(c,:);
                gbInput = gbInput + backprop(c, :)';
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1 % ����������ز�֮��Ĳ�������һ��һ�㽫��Ӧ�ݶ�ֵ�Ž�g��
    gw = zeros(size(w)); % ��ʼ�����ص�w���ݶ�gw
    gw(1:nVars*nHidden(1)) = gwInput(:); % �����w�ݶ�
    offset = nVars * nHidden(1); % һ��ָ�룬������һ����ݶȷŵ�����
    for h = 2:length(nHidden)
        gw(offset+1:offset+nHidden(h-1)*nHidden(h)) = gwHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    gw(offset+1:offset+nHidden(end)*nLabels) = gwOutput(:); % �����w�ݶ�
    
    gb = zeros(size(b)); % ��ʼ�����ص�b���ݶ�gb
    gb(1:nHidden(1)) = gbInput(:); % �����b�ݶ�
    offset = nHidden(1); % һ��ָ�룬������һ����ݶȷŵ�����
    for h = 2:length(nHidden)
        gb(offset+1:offset+nHidden(h)) = gbHidden{h-1};
        offset = offset + nHidden(h);
    end
    gb(offset+1:offset+nLabels) = gbOutput(:); % �����b�ݶ�
end
end
