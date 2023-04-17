function [distorted_img] = dist_generator(img,dist_type,level,seed)
%% set distortion parameter
    bright_level = [2,3,4,5,6];
    dark_level = [1,1.5,2,2.5,3];
    wn_level = [-10,-8.5,-7.5,-6.5,-5.5];
    mot_level = [ 6,	10.5,	15,	19.5,	24];
    cont_level = [0.11,	0.15, 0.20,	0.27,	0.35];
    gblur_level = [5,	8,	11,	14,	17];
    ca_level = [4,	7,	10,	13,	16];
    jpeg_level = [43,35,28,21,15];
    jp2k_level = [0.46,0.36,0.26,0.16,0.06];
    veg_level = [10, 9.2, 8.3, 7.5, 6.1];
   %% distortion generation
    rng(seed)
    switch dist_type
        case 0 
            distorted_img = img;
        case 1 %overexposure
            g = bright_level(level);
            img = double(img);
            img = img/255;
            distorted_img = min(1.0, img + 0.1*g);  % Shifted
            distorted_img = round(255*distorted_img);
            distorted_img = uint8(distorted_img);
        case 2 %underexposure
            g = dark_level(level);
            img = double(img);
            img = img/255;
            distorted_img = min(1.0, img - 0.1*g);  % Shifted
            distorted_img = round(255*distorted_img);
            distorted_img = uint8(distorted_img);
        case 3 %gaussian noise
            rng(seed);
            distorted_img = imnoise(img,'gaussian',0,2^(wn_level(level)));
        case 4 %motion blur
            mot_angle = randi([0,360]);
            PSF = fspecial('motion',mot_level(1,level),mot_angle);
            distorted_img = imfilter(img, PSF, 'conv', 'symmetric', 'same');
        case 5 % out of focus
            hsize = gblur_level(level);
            h = fspecial('gaussian', hsize, hsize/6);
            distorted_img = imfilter(img,h,'symmetric', 'same');
        case 6 % chromatic aberration
            hsize = 3;
            R=(img(:,:,1));
            G=(img(:,:,3));
            B=(img(:,:,2));
            R2=R;
            B2=B;
            R2(:,ca_level(level):end)=R(:,1:end-ca_level(level)+1);
            B2(:,ca_level(level)/2:end)=B(:,1:end-ca_level(level)/2+1);
            temp = img;
            temp(:,:,1)=R2;
            temp(:,:,2)=B2;
            temp_img = temp;
            h = fspecial('gaussian', hsize, hsize/6);
            distorted_img=imfilter(temp_img,h,'symmetric');
        case 7 % contrast change
            distorted_img = img;
            for chann = 1:3
                I = img(:,:,chann);
                distorted_img(:,:,chann) = imadjust(I,[],[cont_level(level), 1-cont_level(level)]);
            end
        case 8 % JPEG 
            testName = [num2str(randi(intmax)) '.jpg'];
            imwrite(img, testName,'jpg','quality',jpeg_level(level));
            distorted_img = imread(testName);
            delete(testName);
        case 9 %JP2K(JPEG2000 compression)
            testName = [num2str(randi(intmax)) '.jp2'];
            imwrite(img,testName,'jp2','CompressionRatio', 24 / jp2k_level(level));
            distorted_img = imread(testName);
            delete(testName);  
        case 10 % vignetting
            MAX_ANGLE = 3.14159/veg_level(level);
            [height, width, ~] = size(img);
            center = [randi(height) randi(width)];
            max_img_rad = sqrt((height/2)^2+(width/2)^2);
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
    end
end