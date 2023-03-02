function distorted_img = imghost(img, img1, level)

img = double(img);
img1 = double(img1);
h = size(img,1);
w = size(img,2);

new_img1=imresize(img1,[h w]);
distorted_img(:,:,1) = round(level*img(:,:,1)+(1-level)*new_img1(:,:,1));
distorted_img(:,:,2) = round(level*img(:,:,2)+(1-level)*new_img1(:,:,2));
distorted_img(:,:,3) = round(level*img(:,:,3)+(1-level)*new_img1(:,:,3));
distorted_img = uint8(distorted_img);

end
