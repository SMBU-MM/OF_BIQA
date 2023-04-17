clear all
clc
close all
seed=1
rng(seed)
MAX_ANGLE = 3.14159/10;
img = imread("Img_1.png");
[height, width, ~] = size(img);
center = [randi(height) randi(width)];
max_img_rad = sqrt((center(1))^2+(center(2))^2);
eikona = [];
for x=1:height
    for y=1:width
        eikona(x,y) = (cos((sqrt((center(1)-x)^2 + (center(2)-y)^2)/max_img_rad)*MAX_ANGLE))^4;
    end
end
new_eikona(:,:,1)= eikona;
new_eikona(:,:,2)= eikona;
new_eikona(:,:,3)= eikona;
distorted_img = uint8(double(img).*new_eikona);
imshow(distorted_img)