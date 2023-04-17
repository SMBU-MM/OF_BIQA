clear all
close all
clc
warning off
%% distortions 
dist_names = ["OE", "UE", "GN", "MB", "OF", "CA", "CC", "JPEG", "JP2K", "VG"];
fid = fopen('image_names.txt','w');

count = 1;
for img_idx=1:4744
    t0 = cputime;
    path = 'E:\\domain_adaptation\\exploration_database_and_code\\pristine_images\\';
    img = imread(strcat(path, num2str(img_idx,'%05d'),'.bmp'));
    for dist_com_type = 1:4
        if dist_com_type==1
            for dist_type=1:10
                rng(count)
                dist_levels = randperm(5, 2);
                for idx = 1:2
                    [distorted_img] = dist_generator(img,dist_type,dist_levels(idx),count*2);
                    imwrite(distorted_img, strcat('./sim/', num2str(count, '%08d'), '.png'));
                    fprintf(fid,'%d,%s,%s,%s,%d\n',dist_com_type, strcat(num2str(img_idx,'%05d'),'.bmp'),strcat(num2str(count, '%08d'), '.png'),...
                        dist_names(dist_type), dist_levels(idx));
                    count = count + 1; 
                end
            end
        elseif dist_com_type==2
            % two distortions
            rng(count)
            dist_types = nchoosek([1:10],2);
            dist_idxs = randperm(length(dist_types),15);
            for idx=1:15
                rng(count)
                dist_level = randperm(5, 1);
                [distorted_img] = dist_generator(img,dist_types(dist_idxs(idx), 1),dist_level,count*2);
                rng(count*2)
                dist_level1 = randperm(5, 1);
                [distorted_img] = dist_generator(distorted_img,dist_types(dist_idxs(idx), 2),dist_level1,count*2);
                imwrite(distorted_img, strcat('./sim/', num2str(count, '%08d'), '.png'));
                fprintf(fid,'%d,%s,%s,%s,%s\n',dist_com_type, strcat(num2str(img_idx,'%05d'),'.bmp'),strcat(num2str(count, '%08d'), '.png'),...
                        strcat(dist_names(dist_types(dist_idxs(idx), 1)),'+' ,dist_names(dist_types(dist_idxs(idx), 2))), ...
                        strcat(num2str(dist_level),'+', num2str(dist_level1)));
                count = count + 1;
            end
        elseif dist_com_type==3
            % three distortions
             rng(count)
             dist_types = nchoosek([1:10],3);
             dist_idxs = randperm(length(dist_types),10);
             for idx=1:10
                rng(count)
                dist_level = randperm(5, 1);
                [distorted_img] = dist_generator(img,dist_types(dist_idxs(idx), 1),dist_level,count*2);
                rng(count*2)
                dist_level1 = randperm(5, 1);
                [distorted_img] = dist_generator(distorted_img,dist_types(dist_idxs(idx), 2),dist_level1,count*2);
                rng(count/2)
                dist_level2 = randperm(5, 1);
                [distorted_img] = dist_generator(distorted_img,dist_types(dist_idxs(idx), 3),dist_level2,count*2);
                
                imwrite(distorted_img, strcat('./sim/', num2str(count, '%08d'), '.png'));
                fprintf(fid,'%d,%s,%s,%s,%s\n',dist_com_type, strcat(num2str(img_idx,'%05d'),'.bmp'),strcat(num2str(count, '%08d'), '.png'),...
                        strcat(dist_names(dist_types(dist_idxs(idx), 1)),'+', dist_names(dist_types(dist_idxs(idx), 2)),'+', dist_names(dist_types(dist_idxs(idx), 3))),  ...
                        strcat(num2str(dist_level),'+', num2str(dist_level1), '+', num2str(dist_level2)));
                count = count + 1;
            end
        else
            % four distortions
            rng(count)
            dist_types = nchoosek([1:10],4);
            dist_idxs = randperm(length(dist_types),5);
            for idx=1:5
                rng(count)
                dist_level = randperm(5, 1);
                [distorted_img] = dist_generator(img,dist_types(dist_idxs(idx), 1),dist_level,count*2);
                rng(count*2);
                dist_level1 = randperm(5, 1);
                [distorted_img] = dist_generator(distorted_img,dist_types(dist_idxs(idx), 2),dist_level1,count*2);
                rng(count*3)
                dist_level2 = randperm(5, 1);
                [distorted_img] = dist_generator(distorted_img,dist_types(dist_idxs(idx), 3),dist_level2,count*2);
                rng(count*4)
                dist_level3 = randperm(5, 1);
                [distorted_img] = dist_generator(distorted_img,dist_types(dist_idxs(idx), 4),dist_level3,count*2);
                imwrite(distorted_img, strcat('./sim/', num2str(count, '%08d'), '.png'));
                fprintf(fid,'%d,%s,%s,%s,%s\n',dist_com_type, strcat(num2str(img_idx,'%05d'),'.bmp'),strcat(num2str(count, '%08d'), '.png'),...
                      strcat(dist_names(dist_types(dist_idxs(idx), 1)),'+', dist_names(dist_types(dist_idxs(idx), 2)),'+', dist_names(dist_types(dist_idxs(idx), 3)), '+', dist_names(dist_types(dist_idxs(idx), 3))),  ...
                      strcat(num2str(dist_level),'+', num2str(dist_level1), '+', num2str(dist_level2), '+', num2str(dist_level3)));
                count = count + 1;
            end
        end
    end
    disp(strcat('completed:', num2str(img_idx), '     time:',  num2str(cputime-t0)))
end

