fid=fopen('preds_3.txt','w');
fsimval = 0;
sr_sim = 0;
nlpds = 0;
vsi_value = 0;
gmsd_value = 0;
count = 1;
for i = 1:237200
    dist_name = strcat(num2str(i, '%08d'), '.png'); 
    ref_name = strcat(num2str(count, '%05d'), '.bmp');
    ref_path = strcat('../sim/', ref_name);
    dist_path = strcat("../sim/", dist_name);
    I1 = imread(ref_path);
    I2 = imread(dist_path);
    [~,fsimval(i,1)] = FeatureSIM(I2, I1);
    sr_sim(i,1) = SR_SIM(I2, I1);
    [nlpds(i,1),  ~] = NLP_dist(double(rgb2gray(I1))/255.0,double(rgb2gray(I2))/255.0);
    vsi_value(i,1) = VSI(I2, I1);
    [gmsd_value(i,1), ~] = GMSD(I1, I2)
    fprintf(fid,  "%s,%s,%f,%f,%f,%f,%f\n", ref_name, dist_name, fsimval(i,1), sr_sim(i,1), nlpds(i,1), vsi_value(i,1), gmsd_value(i,1) );
    if mod(i,10)==0
        disp(['have completed:', num2str(i)])
    end
    if mod(i,50)==0
        count = count + 1
    end
end
fclose(fid);
