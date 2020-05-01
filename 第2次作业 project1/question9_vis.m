for i=1:5
    image = X(i+5000,2:257); % ¿ÉĞŞ¸ÄÎªi+5000 / i+10000 / i+15000 / i+20000
    Min = min(image);
    Max = max(image);
    image = (image - Min) / (Max - Min);
    image = reshape(image, 16, 16);
    image = imresize(image, [320, 320]);
    imshow(image);
    pause(0.5);
end