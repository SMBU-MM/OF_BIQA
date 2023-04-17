
%
% This function performs the image quality distance explained in:
% "Perceptual Image Quality Assessment Using Laplacian Pyramid with Divisive Normalization"
% V. Laparra, J. Ballé, A. Berardino, and E. P. Simoncelli
%
% [DMOS_Lap_dn2 DMOS_Lap] = NLP_Distance(IM_1,IM_2,DN_filts)
%
% Inputs:
%  IM_1 = Original image in gray scale image (between 0 and 1)
%  IM_2 = Distorted image in gray scale image (between 0 and 1)
%  DN_filts (optional) = struct containing the parameters for the divisive
%  normalization. Default = the ones used in the paper. 
%
% Outputs:
%  DMOS_Lap_dn2 = Predicted distance in the normalized domain
%  DMOS_Lap = Predicted distance in the Laplacian domain (for illustration
%  purposes)
%

function [DMOS_Lap_dn2 DMOS_Lap] = NLP_dist(IM_1,IM_2,DN_filts)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('DN_filts')
    DN_filts = DN_filters;
end

N_levels = 6;

[Y_ori Lap_ori] = NLP(IM_1,DN_filts);
[Y_dist Lap_dist] = NLP(IM_2,DN_filts);

for N_b = 1:N_levels
    RR_Lap_aux(N_b) = sqrt(mean((Lap_ori{N_b}(:)-Lap_dist{N_b}(:)).^2));
    RR_aux(N_b) = sqrt(mean((Y_ori{N_b}(:)-Y_dist{N_b}(:)).^2));
end

DMOS_Lap = mean(RR_Lap_aux);
DMOS_Lap_dn2 = mean(RR_aux);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXTRA FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NLP Transformation
function [DN_dom Lap_dom] = NLP(IM,DN_filts)

N_levels = 6;

Lap_dom = laplacian_pyramid_s(IM,N_levels);

for N_b = 1:N_levels
    
    A2 = conv2(abs(Lap_dom{N_b}),DN_filts(N_b).F2,'same');
    DN_dom{N_b} = Lap_dom{N_b} ./ (DN_filts(N_b).sigma + A2);

end

%% Parameters for the divisive normalization

function DN_filts = DN_filters

% These parameters were learned using the McGill dataset
% Training_NLP_param.m

sigmas = [0.0248    0.0185    0.0179    0.0191    0.0220    0.2782];

for N_b = 1:6
    DN_filts(N_b).sigma = sigmas(N_b);
end


DN_filts(1).F2 = [0         0         0         0         0
         0         0    0.1011         0         0
         0    0.1493         0    0.1460    0.0072
         0         0    0.1015         0         0
         0         0         0         0         0];


DN_filts(2).F2(2:4,2:4) = [     0    0.0757         0
                           0.1986         0    0.1846
                                0    0.0837         0];
DN_filts(2).F2(5,5) = 0;
         
DN_filts(3).F2(2:4,2:4) = [      0    0.0477         0
                            0.2138         0    0.2243
                                 0    0.0467         0];
DN_filts(3).F2(5,5) = 0;         
         
DN_filts(4).F2(2:4,2:4) = [ 0    0     0;         
                        0.2503   0    0.2616;         
                            0    0    0];
DN_filts(4).F2(5,5) = 0;

DN_filts(5).F2(2:4,2:4) = [0     0    0;   
                        0.2598   0    0.2552;         
                           0    0    0];
DN_filts(5).F2(5,5) = 0;


DN_filts(6).F2(2:4,2:4) = [ 0    0   0;
                        0.2215   0  0.0717;         
                            0    0   0];
DN_filts(6).F2(5,5) = 0;


%% Contruction of Laplacian pyramid
%
% Arguments:
%   image 'I'
%   'nlev', number of levels in the pyramid (optional)
%
% tom.mertens@gmail.com, August 2007
%
%
% More information:
%   'The Laplacian Pyramid as a Compact Image Code'
%   Burt, P., and Adelson, E. H., 
%   IEEE Transactions on Communication, COM-31:532-540 (1983). 
%


function pyr = laplacian_pyramid_s(I,nlev) 
 
r = size(I,1); 
c = size(I,2); 
 
if ~exist('nlev') 
    % compute the highest possible pyramid     
    nlev = floor(log(min(r,c)) / log(2)); 
end 
 
% recursively build pyramid 
pyr = cell(nlev,1); 
f = [.05, .25, .4, .25, .05];  % original [Burt and Adelson, 1983]
filter = f'*f;

J = I; 
for l = 1:nlev - 1 
    % apply low pass filter, and downsample 
    I = downsample(J,filter); 
    odd = 2*size(I) - size(J);  % for each dimension, check if the upsampled version has to be odd 
    % in each level, store difference between image and upsampled low pass version 
    pyr{l} = J - upsample(I,odd,filter); 
    J = I; % continue with low pass image 
end 
pyr{nlev} = J; % the coarest level contains the residual low pass image 


function R = downsample(I, filter) 
 
% low pass, convolve with filter 
R = imfilter(I,filter,'symmetric');    
 
% decimate 
r = size(I,1); 
c = size(I,2); 
R = R(1:2:r, 1:2:c, :);  
 
function R = upsample(I,odd,filter) 
 
% increase resolution 
I = padarray(I,[1 1 0],'replicate'); % pad the image with a 1-pixel border 
r = 2*size(I,1); 
c = 2*size(I,2); 
k = size(I,3); 
R = zeros(r,c,k); 
R(1:2:r, 1:2:c, :) = 4*I; % increase size 2 times; the padding is now 2 pixels wide 
 
% interpolate, convolve with filter 
R = imfilter(R,filter);     
 
% remove the border 
R = R(3:r - 2 - odd(1), 3:c - 2 - odd(2), :);





