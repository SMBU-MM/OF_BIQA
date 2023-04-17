# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:20:33 2020

@author: Administrator
"""
import os, torch, random, math
from PIL import Image
import numpy as np
from utils import prepare_image
from IQA_pytorch import GMSD
from itertools import combinations
from scipy.optimize import curve_fit
import copy
from mdsi import mdsi
import scipy.stats
import cv2
from DISTS_pytorch import DISTS


random.seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GMSD_D = GMSD(channels=3).to(device)
DISTS_D = DISTS().to(device)
def bt_model(x, y, lists, temp):
   exp = math.exp((x - y)/(abs(max(lists) -min(lists))*temp))
   label = exp/( 1.0 + exp)
   return label

if __name__ == '__main__':
    with open('pseduo_label.txt', 'w') as pseduo_label:
        with open('preds.txt', 'r') as ssim_vsi_nlpd_file:
            ssim_vsi_nlpd = ssim_vsi_nlpd_file.readlines()
            for step, line in enumerate(ssim_vsi_nlpd):
                list_temp = line.split(',')
                img_name = list_temp[1]
                ref_name = list_temp[0]
    
                dist_img = prepare_image(Image.open(os.path.join('../sim', img_name)).convert("RGB")).to(device)
                ref_img = prepare_image(Image.open(os.path.join('../sim', ref_name)).convert("RGB")).to(device)
        
                fsim = float(list_temp[2])
                srsim = float(list_temp[3])
                nlpds = -float(list_temp[4])
                vsi = float(list_temp[5].replace('\n', ''))
                mdsi_value = -mdsi(os.path.join('../sim', ref_name), os.path.join('../sim', img_name))
                gmsd = -GMSD_D(dist_img, ref_img, as_loss=False).item()
                pseduo_label.write("{},{},{},{},{},{},{},{}\n".format(img_name.replace('\n', ''), ref_name.replace('\n', ''), \
                                                                fsim, srsim, nlpds, vsi, mdsi_value, gmsd))
                if step % 10 == 0:
                    print('have completed:', step)
                    
    
#if __name__ == '__main__':
#    temp = 1.0
#    with open('train_pair_{}_1.txt'.format(temp), 'w') as train_file:
#        with open('pseduo_label.txt', 'r') as wfile:
#            lines = wfile.readlines()
#            image_name = []
#            ref_names = []
#            ssim_list = []
#            gmsd_list = []
#            mdsi_list = []
#            vsi_list = []
#            lpips_list = []
#            nlpds_list = []
#            for line in lines:
#                line_list = line.replace('\n', '').split(',')
#                image_name.append(line_list[0])
#                ref_names.append(line_list[1])
#                ssim_list.append(float(line_list[2]))
#                gmsd_list.append(float(line_list[3]))
#                mdsi_list.append(float(line_list[4]))
#                vsi_list.append(float(line_list[5]))
#                lpips_list.append(float(line_list[6]))
#                nlpds_list.append(float(line_list[7]))
#                lens = len(lines)
#        # same reference image and distortion types, but different distortion level
#        for i in range(0, lens, 5):
#            combs = list(combinations([0,1,2,3,4], 2))
#            random.shuffle(combs)
#            for j in range(0,5):
#                ref_name1 = ref_names[i+ combs[j][0]]
#                img_name1 = image_name[i+ combs[j][0]]
#                ref_name2 = ref_names[i+ combs[j][1]]
#                img_name2 = image_name[i+ combs[j][1]]
#                ssim_label = bt_model(ssim_list[i+ combs[j][0]], ssim_list[i+ combs[j][1]], ssim_list, temp)
#                gmsd_label = bt_model(gmsd_list[i+ combs[j][0]], gmsd_list[i+ combs[j][1]], gmsd_list, temp)
#                mdsi_label = bt_model(mdsi_list[i+ combs[j][0]], mdsi_list[i+ combs[j][1]], mdsi_list, temp)
#                vsi_label = bt_model(vsi_list[i+ combs[j][0]], vsi_list[i+ combs[j][1]], vsi_list, temp)
#                lpips_label = bt_model(lpips_list[i+ combs[j][0]], lpips_list[i+ combs[j][1]], lpips_list, temp)
#                nlpd_label = bt_model(nlpds_list[i+ combs[j][0]], nlpds_list[i+ combs[j][1]], nlpds_list, temp)
#                if ssim_label>0.5 and gmsd_label>0.5 and mdsi_label>0.5 and vsi_label>0.5 and \
#                    lpips_label>0.5 and nlpd_label>0.5:
#                    #print(ssim_label, gmsd_label, mdsi_label, vsi_label, lpips_label, nlpd_label)
#                    train_file.write("{},{},{},{},{},{},{},{},{},{},0\n".format(img_name1, ref_name1, img_name2, ref_name2, ssim_label, gmsd_label, \
#                                                                            mdsi_label, vsi_label, lpips_label, nlpd_label))
#        # same reference image,but different different distortion level
#        for i in range(0, lens, 5*16):
#            dist_types = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2))
#            random.shuffle(dist_types)
#            for dist_type in dist_types[:80]:
#                dist_level = list(combinations([0,1,2,3,4], 2))
#                random.shuffle(dist_level)
#                idx1 = i+ dist_type[0]*5 + dist_level[0][0]
#                idx2 = i+ dist_type[1]*5 + dist_level[0][1]
#                ref_name1 = ref_names[idx1]
#                img_name1 = image_name[idx1]
#                ref_name2 = ref_names[idx2]
#                img_name2 = image_name[idx2]
#
#                ssim_label = bt_model(ssim_list[idx1], ssim_list[idx2], ssim_list, temp)
#                gmsd_label = bt_model(gmsd_list[idx1], gmsd_list[idx2], gmsd_list, temp)
#                mdsi_label = bt_model(mdsi_list[idx1], mdsi_list[idx2], mdsi_list, temp)
#                vsi_label = bt_model(vsi_list[idx1], vsi_list[idx2], vsi_list, temp)
#                lpips_label = bt_model(lpips_list[idx1], lpips_list[idx2], lpips_list, temp)
#                nlpd_label = bt_model(nlpds_list[idx1], nlpds_list[idx2], nlpds_list, temp)
#                if (ssim_label>0.5 and gmsd_label>0.5 and mdsi_label>0.5 and vsi_label>0.5 and \
#                    lpips_label>0.5 and nlpd_label>0.5) or (ssim_label<0.5 and gmsd_label<0.5 and mdsi_label<0.5 and vsi_label<0.5 and \
#                    lpips_label<0.5 and nlpd_label<0.5):
#                    #print(ssim_label, gmsd_label, mdsi_label, vsi_label, lpips_label, nlpd_label)
#                    train_file.write("{},{},{},{},{},{},{},{},{},{},1\n".format(img_name1, ref_name1, img_name2, ref_name2, ssim_label, gmsd_label, \
#                                                                            mdsi_label, vsi_label, lpips_label, nlpd_label))
#        # different reference image, distortion types, distortion levels
#        idx_list = []
#        nums = np.linspace(0,lens,lens, endpoint=False).astype(np.int32).tolist()
#
#        counts = 0
#        while counts < 170000:
#            a = random.choice(nums)
#            b = random.choice(nums)
#            if a<b:
#                tup = (a,b)
#            elif a==b:
#                continue
#            else:
#                tup = (b,a)
#            if tup in idx_list:
#                continue
#            else:
#                idx_list.append(tup)
#                counts += 1
#                if counts % 1000 == 0:
#                    print('have completed:', counts)
#        np.save('idx_list.npy',np.array(idx_list))
#        for step, (i,j) in enumerate(idx_list):
#            pre_img = lines[i].replace('\n', '').split(',')
#            lat_img = lines[j].replace('\n', '').split(',')
#            img_ref_name1 = pre_img[:1]
#            img_ref_name2 = lat_img[:1]
#
#            exp = math.exp((float(pre_img[2]) - float(lat_img[2]))/(abs(max(ssim_list) -min(ssim_list))*temp))
#            ssim_label = exp/( 1.0 + exp)
#            exp = math.exp((float(pre_img[3]) - float(lat_img[3]))/(abs(max(gmsd_list) -min(gmsd_list))*temp))
#            gmsd_label = exp/(1.0 + exp)
#            exp = math.exp((float(pre_img[4]) - float(lat_img[4]))/(abs(max(mdsi_list) -min(mdsi_list))*temp))
#            mad_label = exp/(1.0 + exp)
#            exp = math.exp((float(pre_img[5]) - float(lat_img[5]))/(abs(max(vsi_list) -min(vsi_list))*temp))
#            vsi_label = exp/(1.0 + exp)
#            exp = math.exp((float(pre_img[6]) - float(lat_img[6]))/(abs(max(lpips_list) -min(lpips_list))*temp))
#            lpips_label = exp/(1.0 + exp)
#            exp = math.exp((float(pre_img[7]) - float(lat_img[7]))/(abs(max(nlpds_list) -min(nlpds_list))*temp))
#            nlpds_label = exp/(1.0 + exp)
#            if (ssim_label>0.5 and gmsd_label>0.5 and mdsi_label>0.5 and vsi_label>0.5 and \
#                    lpips_label>0.5 and nlpd_label>0.5) or (ssim_label<0.5 and gmsd_label<0.5 and mdsi_label<0.5 and vsi_label<0.5 and \
#                    lpips_label<0.5 and nlpd_label<0.5):
#                    train_file.write("{},{},{},{},{},{},{},{},{},{},2\n".format(pre_img[0], pre_img[1], lat_img[0], lat_img[1], ssim_label, gmsd_label, \
#                                                                        mad_label, vsi_label, lpips_label, nlpds_label))
#
#        # one is reference image, another is distorted image
#        for i in range(0, lens, 5):
#            combs = list(combinations([0,1,2,3,4], 1))
#            random.shuffle(combs)
#            for j in range(0,5):
#                ref_name1 = ref_names[i+ combs[j][0]]
#                img_name1 = ref_names[i+ combs[j][0]]
#                ref_name2 = ref_names[i+ combs[j][0]]
#                img_name2 = image_name[i+ combs[j][0]]
#                ssim_label = bt_model(max(ssim_list), ssim_list[i+ combs[j][0]], ssim_list, temp)
#                gmsd_label = bt_model(max(gmsd_list), gmsd_list[i+ combs[j][0]], gmsd_list, temp)
#                mdsi_label = bt_model(max(mdsi_list), mdsi_list[i+ combs[j][0]], mdsi_list, temp)
#                vsi_label = bt_model(max(vsi_list), vsi_list[i+ combs[j][0]], vsi_list, temp)
#                lpips_label = bt_model(max(lpips_list), lpips_list[i+ combs[j][0]], lpips_list, temp)
#                nlpd_label = bt_model(max(nlpds_list), nlpds_list[i+ combs[j][0]], nlpds_list, temp)
#                if ssim_label>0.5 and gmsd_label>0.5 and mdsi_label>0.5 and vsi_label>0.5 and \
#                    lpips_label>0.5 and nlpd_label>0.5:
#                    #print(ssim_label, gmsd_label, mdsi_label, vsi_label, lpips_label, nlpd_label)
#                    train_file.write("{},{},{},{},{},{},{},{},{},{},3\n".format(img_name1, ref_name1, img_name2, ref_name2, ssim_label, gmsd_label, \
#                                                                            mdsi_label, vsi_label, lpips_label, nlpd_label))
#        #

            
# if __name__ == '__main__':
#     temp =0.2
#     with open('train_pair_{}_random_sample.txt'.format(temp), 'w') as train_file:
#         with open('pseduo_label.txt', 'r') as wfile:
#             lines = wfile.readlines()
#             ssim_list = []
#             gmsd_list = []
#             mdsi_list = []
#             vsi_list = []
#             lpips_list = []
#             nlpds_list = []
#             for line in lines:
#                 line_list = line.replace('\n', '').split(',')
#                 ssim_list.append(float(line_list[2]))
#                 gmsd_list.append(float(line_list[3]))
#                 mdsi_list.append(float(line_list[4]))
#                 vsi_list.append(float(line_list[5]))
#                 lpips_list.append(float(line_list[6]))
#                 nlpds_list.append(float(line_list[7]))
#             idx_list = []
#             lens = len(lines)
#             nums = np.linspace(0,lens,lens, endpoint=False).astype(np.int32).tolist()
            
#             counts = 0
#             while counts < 400000:
#                 a = random.choice(nums)
#                 b = random.choice(nums)
#                 if a<b:
#                     tup = (a,b)
#                 elif a==b:
#                     continue
#                 else:
#                     tup = (b,a)
#                 if tup in idx_list:
#                     continue
#                 else:
#                     idx_list.append(tup)
#                     counts += 1
#                     if counts % 1000 == 0:
#                         print('have completed:', counts)
#             np.save('idx_list.npy',np.array(idx_list))
#             # idx_list = list(combinations(np.linspace(0,lens,lens, endpoint=False).astype(np.int16).tolist(), 2))
#             # idx_list = random.sample(idx_list, 400000)
        
#             for step, (i,j) in enumerate(idx_list):
#                 pre_img = lines[i].replace('\n', '').split(',')
#                 lat_img = lines[j].replace('\n', '').split(',')
#                 img_ref_name1 = pre_img[:1]
#                 img_ref_name2 = lat_img[:1]

#                 exp = math.exp((float(pre_img[2]) - float(lat_img[2]))/abs(max(ssim_list) -min(ssim_list)*temp))
#                 ssim_label = exp/( 1.0 + exp)
#                 exp = math.exp((float(pre_img[3]) - float(lat_img[3]))/abs(max(gmsd_list) -min(gmsd_list)*temp))
#                 gmsd_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[4]) - float(lat_img[4]))/abs(max(mdsi_list) -min(mdsi_list)*temp))
#                 mdsi_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[5]) - float(lat_img[5]))/abs(max(vsi_list) -min(vsi_list)*temp))
#                 vsi_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[6]) - float(lat_img[6]))/abs(max(lpips_list) -min(lpips_list)*temp))
#                 lpips_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[7]) - float(lat_img[7]))/abs(max(nlpds_list) -min(nlpds_list)*temp))
#                 nlpds_label = exp/(1.0 + exp)
#                 if (ssim_label>0.5 and gmsd_label>0.5 and mdsi_label>0.5 and vsi_label>0.5 and \
#                    lpips_label>0.5 and nlpds_label>0.5) or (ssim_label<0.5 and gmsd_label<0.5 and mdsi_label<0.5 and vsi_label<0.5 and \
#                    lpips_label<0.5 and nlpds_label<0.5):
                
#                    train_file.write("{},{},{},{},{},{},{},{},{},{},1\n".format(pre_img[0], pre_img[1], lat_img[0], lat_img[1], ssim_label, gmsd_label, \
#                                                                              mdsi_label, vsi_label, lpips_label, nlpds_label))

# #####################################################################################################################
# # nonlinear fitting
# #####################################################################################################################
# def four_params_func(x, eta1, eta2, eta3, eta4):
#     return (eta1-eta2)/(1+np.exp(-(x -eta3)/np.abs(eta4))) + eta2

# def inv_four_params_func(y, eta1, eta2, eta3, eta4):
#     return eta3 - np.abs(eta4)*np.log((eta1-y)/(y-eta2))

# def nonlinearfit(xdata, ydata):
#     p0 = np.array([np.max(ydata), np.min(ydata), np.mean(xdata), 0.5])
#     srcc = scipy.stats.mstats.spearmanr( x=xdata, y=ydata)[0]
#     krcc = scipy.stats.mstats.kendalltau(x=xdata, y=ydata)[0]
#     popt, _= curve_fit(four_params_func, xdata, ydata, p0=p0, maxfev=500000)
#     ypre = four_params_func(xdata, *popt)
#     plcc = scipy.stats.pearsonr( x=ydata,  y=ypre)[0]
#     rmse = np.sqrt(np.sum(np.power(ypre-ydata, 2)))/len(ydata)
    
#     return popt, srcc, plcc, rmse, krcc

# def test():
#     with open('live_file_name.txt', 'r') as live_img_ref_file:
#         live_img_refs = live_img_ref_file.readlines()
#         names_list = []
#         ref_list = []
#         for line in live_img_refs:
#             list_temp = line.split(',')
#             names_list.append(list_temp[0])
#             ref_list.append(list_temp[1])
#     srcc = {'live':{}, 'csiq':{}, 'tid2013':{}, 'kadid10k':{}}
#     plcc = {'live':{}, 'csiq':{}, 'tid2013':{}, 'kadid10k':{}}
    
#     # ##################################################################################################################################
#     # # LIVE
#     # ##################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('databaserelease2/splits2/1/live_test_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('databaserelease2/splits2/1/live_test.txt', 'r') as file:
#         lines = file.readlines()
#         mos_list = []
#         ssim_list = []
#         gmsd_list = []
#         mad_list = []
#         vsi_list = []
#         lpips_list = []
#         nlpds_list = []
#         for step, line in enumerate(lines):
#             line_list = line.split('\t')
#             img_name = line_list[0]
#             ref_name = 'refimgs/' + ref_list[names_list.index(img_name)]

#             print(img_name, ref_name)

#             mos = float(line_list[1].replace('\n', ''))
#             mos_list.append(mos)
#             std = line_list[2].replace('\n', '')
#             dist_img = prepare_image(Image.open(os.path.join('databaserelease2', img_name)).convert("RGB")).to(device)
#             ref_img = prepare_image(Image.open(os.path.join('databaserelease2', ref_name)).convert("RGB")).to(device)
#             # print(img_name, names_matlab[step])
#             ssim_list.append(ssim_matlab[step])
#             gmsd_list.append(-GMSD_D(dist_img, ref_img, as_loss=False).item())
#             mad_list.append(-MAD_D(dist_img, ref_img, as_loss=False).item())
#             vsi_list.append(vsi_matlab[step])
#             lpips_list.append(-LPIPS_D(dist_img, ref_img, as_loss=False).item())
#             nlpds_list.append(-nlpd_matlab[step])
            
#         srcc['live']['ssim'] = scipy.stats.mstats.spearmanr(x=mos_list, y=ssim_list)[0]
#         srcc['live']['gmsd'] = scipy.stats.mstats.spearmanr(x=mos_list, y=gmsd_list)[0]
#         srcc['live']['mad'] = scipy.stats.mstats.spearmanr(x=mos_list, y=mad_list)[0]
#         srcc['live']['vsi'] = scipy.stats.mstats.spearmanr(x=mos_list, y=vsi_list)[0]
#         srcc['live']['lpips'] = scipy.stats.mstats.spearmanr(x=mos_list, y=lpips_list)[0]
#         srcc['live']['nlpds'] = scipy.stats.mstats.spearmanr(x=mos_list, y=nlpds_list)[0]
        
#         plcc['live']['ssim'] = scipy.stats.mstats.pearsonr(x=mos_list, y=ssim_list)[0]
#         plcc['live']['gmsd'] = scipy.stats.mstats.pearsonr(x=mos_list, y=gmsd_list)[0]
#         plcc['live']['mad'] = scipy.stats.mstats.pearsonr(x=mos_list, y=mad_list)[0]
#         plcc['live']['vsi'] = scipy.stats.mstats.pearsonr(x=mos_list, y=vsi_list)[0]
#         plcc['live']['lpips'] = scipy.stats.mstats.pearsonr(x=mos_list, y=lpips_list)[0]
#         plcc['live']['nlpds'] = scipy.stats.mstats.pearsonr(x=mos_list, y=nlpds_list)[0]
#     ##################################################################################################################################
#     # CSIQ
#     ##################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('CSIQ/splits2/1/csiq_test_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('CSIQ/splits2/1/csiq_test.txt', 'r') as file:
#         lines = file.readlines()
#         mos_list = []
#         ssim_list = []
#         gmsd_list = []
#         mad_list = []
#         vsi_list = []
#         lpips_list = []
#         nlpds_list = []
#         for step, line in enumerate(lines):
#             line_list = line.split('\t')
#             img_name = line_list[0]
#             ref_name = img_name.split("/")[-1].split('.')[0] + '.png'
#             ref_name = 'src_imgs/' + ref_name

#             print(img_name, ref_name)

#             #ref_name = ref_list[names_list.index(img_name)]
#             mos = line_list[1].replace('\n', '')
#             mos_list.append(float(mos))
#             std = line_list[2].replace('\n', '')
#             print(img_name, ref_name)
#             dist_img = prepare_image(Image.open(os.path.join('CSIQ', img_name)).convert("RGB")).to(device)
#             ref_img = prepare_image(Image.open(os.path.join('CSIQ', ref_name)).convert("RGB")).to(device)
    
#             ssim_list.append(ssim_matlab[step])
#             gmsd_list.append(-GMSD_D(dist_img, ref_img, as_loss=False).item())
#             mad_list.append(-MAD_D(dist_img, ref_img, as_loss=False).item())
#             vsi_list.append(vsi_matlab[step])
#             lpips_list.append(-LPIPS_D(dist_img, ref_img, as_loss=False).item())
#             nlpds_list.append(-nlpd_matlab[step])
            
#         srcc['csiq']['ssim'] = scipy.stats.mstats.spearmanr(x=mos_list, y=ssim_list)[0]
#         srcc['csiq']['gmsd'] = scipy.stats.mstats.spearmanr(x=mos_list, y=gmsd_list)[0]
#         srcc['csiq']['mad'] = scipy.stats.mstats.spearmanr(x=mos_list, y=mad_list)[0]
#         srcc['csiq']['vsi'] = scipy.stats.mstats.spearmanr(x=mos_list, y=vsi_list)[0]
#         srcc['csiq']['lpips'] = scipy.stats.mstats.spearmanr(x=mos_list, y=lpips_list)[0]
#         srcc['csiq']['nlpds'] = scipy.stats.mstats.spearmanr(x=mos_list, y=nlpds_list)[0]
        
#         plcc['csiq']['ssim'] = scipy.stats.mstats.pearsonr(x=mos_list, y=ssim_list)[0]
#         plcc['csiq']['gmsd'] = scipy.stats.mstats.pearsonr(x=mos_list, y=gmsd_list)[0]
#         plcc['csiq']['mad'] = scipy.stats.mstats.pearsonr(x=mos_list, y=mad_list)[0]
#         plcc['csiq']['vsi'] = scipy.stats.mstats.pearsonr(x=mos_list, y=vsi_list)[0]
#         plcc['csiq']['lpips'] = scipy.stats.mstats.pearsonr(x=mos_list, y=lpips_list)[0]
#         plcc['csiq']['nlpds'] = scipy.stats.mstats.pearsonr(x=mos_list, y=nlpds_list)[0]
        
#     ##################################################################################################################################
#     # TID2013
#     ##################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('TID2013/splits2/1/tid_test_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('TID2013/splits2/1/tid_test.txt', 'r') as file:
#         lines = file.readlines()
#         mos_list = []
#         ssim_list = []
#         gmsd_list = []
#         mad_list = []
#         vsi_list = []
#         lpips_list = []
#         nlpds_list = []
#         for step, line in enumerate(lines):
#             line_list = line.split('\t')
#             img_name = line_list[0]
            
#             ref_name = (img_name.split("/")[-1].split('_')[0]).upper() + '.BMP'
#             ref_name = 'reference_images/' + ref_name
            
#             print(img_name, ref_name)

#             mos = line_list[1].replace('\n', '')
#             mos_list.append(float(mos))
#             std = line_list[2].replace('\n', '')
#             dist_img = prepare_image(Image.open(os.path.join('TID2013', img_name)).convert("RGB")).to(device)
#             ref_img = prepare_image(Image.open(os.path.join('TID2013', ref_name)).convert("RGB")).to(device)
    
#             ssim_list.append(ssim_matlab[step])
#             gmsd_list.append(-GMSD_D(dist_img, ref_img, as_loss=False).item())
#             mad_list.append(-MAD_D(dist_img, ref_img, as_loss=False).item())
#             vsi_list.append(vsi_matlab[step])
#             lpips_list.append(-LPIPS_D(dist_img, ref_img, as_loss=False).item())
#             nlpds_list.append(-nlpd_matlab[step])
            
#         srcc['tid2013']['ssim'] = scipy.stats.mstats.spearmanr(x=mos_list, y=ssim_list)[0]
#         srcc['tid2013']['gmsd'] = scipy.stats.mstats.spearmanr(x=mos_list, y=gmsd_list)[0]
#         srcc['tid2013']['mad'] = scipy.stats.mstats.spearmanr(x=mos_list, y=mad_list)[0]
#         srcc['tid2013']['vsi'] = scipy.stats.mstats.spearmanr(x=mos_list, y=vsi_list)[0]
#         srcc['tid2013']['lpips'] = scipy.stats.mstats.spearmanr(x=mos_list, y=lpips_list)[0]
#         srcc['tid2013']['nlpds'] = scipy.stats.mstats.spearmanr(x=mos_list, y=nlpds_list)[0]
        
#         plcc['tid2013']['ssim'] = scipy.stats.mstats.pearsonr(x=mos_list, y=ssim_list)[0]
#         plcc['tid2013']['gmsd'] = scipy.stats.mstats.pearsonr(x=mos_list, y=gmsd_list)[0]
#         plcc['tid2013']['mad'] = scipy.stats.mstats.pearsonr(x=mos_list, y=mad_list)[0]
#         plcc['tid2013']['vsi'] = scipy.stats.mstats.pearsonr(x=mos_list, y=vsi_list)[0]
#         plcc['tid2013']['lpips'] = scipy.stats.mstats.pearsonr(x=mos_list, y=lpips_list)[0]
#         plcc['tid2013']['nlpds'] = scipy.stats.mstats.pearsonr(x=mos_list, y=nlpds_list)[0]
        
#     ##################################################################################################################################
#     # Kadid-10k
#     ##################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('kadid10k/splits2/1/kadid10k_test_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
#     with open('kadid10k/splits2/1/kadid10k_test.txt', 'r') as file:
#         lines = file.readlines()
#         mos_list = []
#         ssim_list = []
#         gmsd_list = []
#         mad_list = []
#         vsi_list = []
#         lpips_list = []
#         nlpds_list = []
#         for step, line in enumerate(lines):
#             line_list = line.split('\t')
#             img_name = line_list[0]
#             ref_name = (img_name.split("/")[-1].split('_')[0]).upper() + '.png'
#             ref_name = 'images/' + ref_name
            
#             print(img_name, ref_name)

#             #ref_name = ref_list[names_list.index(img_name)]
#             mos = line_list[1].replace('\n', '')
#             mos_list.append(float(mos))
#             std = line_list[2].replace('\n', '')
#             dist_img = prepare_image(Image.open(os.path.join('kadid10k', img_name)).convert("RGB")).to(device)
#             ref_img = prepare_image(Image.open(os.path.join('kadid10k', ref_name)).convert("RGB")).to(device)
    
#             ssim_list.append(ssim_matlab[step])
#             gmsd_list.append(-GMSD_D(dist_img, ref_img, as_loss=False).item())
#             mad_list.append(-MAD_D(dist_img, ref_img, as_loss=False).item())
#             vsi_list.append(vsi_matlab[step])
#             lpips_list.append(-LPIPS_D(dist_img, ref_img, as_loss=False).item())
#             nlpds_list.append(-nlpd_matlab[step])
            
#         srcc['kadid10k']['ssim'] = scipy.stats.mstats.spearmanr(x=mos_list, y=ssim_list)[0]
#         srcc['kadid10k']['gmsd'] = scipy.stats.mstats.spearmanr(x=mos_list, y=gmsd_list)[0]
#         srcc['kadid10k']['mad'] = scipy.stats.mstats.spearmanr(x=mos_list, y=mad_list)[0]
#         srcc['kadid10k']['vsi'] = scipy.stats.mstats.spearmanr(x=mos_list, y=vsi_list)[0]
#         srcc['kadid10k']['lpips'] = scipy.stats.mstats.spearmanr(x=mos_list, y=lpips_list)[0]
#         srcc['kadid10k']['nlpds'] = scipy.stats.mstats.spearmanr(x=mos_list, y=nlpds_list)[0]
        
#         plcc['kadid10k']['ssim'] = scipy.stats.mstats.pearsonr(x=mos_list, y=ssim_list)[0]
#         plcc['kadid10k']['gmsd'] = scipy.stats.mstats.pearsonr(x=mos_list, y=gmsd_list)[0]
#         plcc['kadid10k']['mad'] = scipy.stats.mstats.pearsonr(x=mos_list, y=mad_list)[0]
#         plcc['kadid10k']['vsi'] = scipy.stats.mstats.pearsonr(x=mos_list, y=vsi_list)[0]
#         plcc['kadid10k']['lpips'] = scipy.stats.mstats.pearsonr(x=mos_list, y=lpips_list)[0]
#         plcc['kadid10k']['nlpds'] = scipy.stats.mstats.pearsonr(x=mos_list, y=nlpds_list)[0]
#     return srcc, plcc



# def train_pred():
#     with open('live_file_name.txt', 'r') as live_img_ref_file:
#         live_img_refs = live_img_ref_file.readlines()
#         names_list = []
#         ref_list = []
#         for line in live_img_refs:
#             list_temp = line.split(',')
#             names_list.append(list_temp[0])
#             ref_list.append(list_temp[1])
            
#     # ##################################################################################################################################
#     # # LIVE
#     # ##################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('databaserelease2/splits2/1/live_train_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('databaserelease2/splits2/1/live_train_score_kd.txt', 'w') as wfile:
#         with open('databaserelease2/splits2/1/live_train_score.txt', 'r') as file:
#             lines = file.readlines()
#             for step, line in enumerate(lines):
#                 line_list = line.split('\t')
#                 img_name = line_list[0]
#                 ref_name = 'refimgs/' + ref_list[names_list.index(img_name)]
#                 mos = line_list[1].replace('\n', '')
#                 std = line_list[2].replace('\n', '')
                
#                 print(img_name, ref_name)
                
#                 dist_img = prepare_image(Image.open(os.path.join('databaserelease2', img_name)).convert("RGB")).to(device)
#                 ref_img = prepare_image(Image.open(os.path.join('databaserelease2', ref_name)).convert("RGB")).to(device)
#                 ssim = ssim_matlab[step]
#                 gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
#                 mad = MAD_D(dist_img, ref_img, as_loss=False).item()
#                 vsi = vsi_matlab[step]
#                 lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
#                 nlpds = nlpd_matlab[step]
#                 wfile.write("{},{},{},{},{},{},{},{},{},{}\n".format(img_name.replace('\n', ''), ref_name.replace('\n', ''), \
#                                                                 mos, std, ssim, gmsd, mad, vsi, lpips, nlpds))

#     ###################################################################################################################################
#     # CSIQ
#     ###################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('CSIQ/splits2/1/csiq_train_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('CSIQ/splits2/1/csiq_train_score_kd.txt', 'w') as wfile:
#         with open('CSIQ/splits2/1/csiq_train_score.txt', 'r') as file:
#             lines = file.readlines()
#             for step, line in enumerate(lines):
#                 line_list = line.split('\t')
#                 img_name = line_list[0]
#                 ref_name = img_name.split("/")[-1].split('.')[0] + '.png'
#                 ref_name = 'src_imgs/' + ref_name
#                 #ref_name = ref_list[names_list.index(img_name)]
#                 mos = line_list[1].replace('\n', '')
#                 std = line_list[2].replace('\n', '')
                
#                 print(img_name, ref_name)
                
#                 dist_img = prepare_image(Image.open(os.path.join('CSIQ', img_name)).convert("RGB")).to(device)
#                 ref_img = prepare_image(Image.open(os.path.join('CSIQ', ref_name)).convert("RGB")).to(device)
        
#                 ssim = ssim_matlab[step]
#                 gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
#                 mad = MAD_D(dist_img, ref_img, as_loss=False).item()
#                 vsi = vsi_matlab[step]
#                 lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
#                 nlpds = nlpd_matlab[step]
#                 wfile.write("{},{},{},{},{},{},{},{},{},{}\n".format(img_name.replace('\n', ''), ref_name.replace('\n', ''), \
#                                                                 mos, std, ssim, gmsd, mad, vsi, lpips, nlpds))

#     ###################################################################################################################################
#     # TID2013
#     ###################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('TID2013/splits2/1/tid_train_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('TID2013/splits2/1/tid_train_score_kd.txt', 'w') as wfile:
#         with open('TID2013/splits2/1/tid_train_score.txt', 'r') as file:
#             lines = file.readlines()
#             for step, line in enumerate(lines):
#                 line_list = line.split('\t')
#                 img_name = line_list[0]
#                 ref_name = (img_name.split("/")[-1].split('_')[0]).upper() + '.BMP'
#                 ref_name = 'reference_images/' + ref_name

#                 print(img_name, ref_name)

#                 #ref_name = ref_list[names_list.index(img_name)]
#                 mos = line_list[1].replace('\n', '')
#                 std = line_list[2].replace('\n', '')
#                 print(img_name, ref_name)
#                 dist_img = prepare_image(Image.open(os.path.join('TID2013', img_name)).convert("RGB")).to(device)
#                 ref_img = prepare_image(Image.open(os.path.join('TID2013', ref_name)).convert("RGB")).to(device)
        
#                 ssim = ssim_matlab[step]
#                 gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
#                 mad = MAD_D(dist_img, ref_img, as_loss=False).item()
#                 vsi = vsi_matlab[step]
#                 lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
#                 nlpds = nlpd_matlab[step]
#                 wfile.write("{},{},{},{},{},{},{},{},{},{}\n".format(img_name.replace('\n', ''), ref_name.replace('\n', ''), \
#                                                                 mos, std, ssim, gmsd, mad, vsi, lpips, nlpds))

#     ###################################################################################################################################
#     # Kadid
#     ###################################################################################################################################
#     print('****************************************************************************************************************************')
#     with open('kadid10k/splits2/1/kadid10k_train_score_matlab_ssim_vsi.txt', 'r') as ssim_vsi_file:
#         ssim_vsi_matlab = ssim_vsi_file.readlines()
#         ssim_matlab = []
#         vsi_matlab = []
#         nlpd_matlab = []
#         names_matlab = []
#         ref_matlab = []
#         for line in ssim_vsi_matlab:
#             list_temp = line.split(',')
#             names_matlab.append(list_temp[0])
#             ref_list.append(list_temp[1])
#             ssim_matlab.append(float(list_temp[4]))
#             vsi_matlab.append(float(list_temp[5]))
#             nlpd_matlab.append(float(list_temp[6]))
            
#     with open('kadid10k/splits2/1/kadid10k_train_score_kd.txt', 'w') as wfile:
#         with open('kadid10k/splits2/1/kadid10k_train_score.txt', 'r') as file:
#             lines = file.readlines()
#             for step, line in enumerate(lines):
#                 line_list = line.split('\t')
#                 img_name = line_list[0]
#                 ref_name = (img_name.split("/")[-1].split('_')[0]).upper() + '.png'
#                 ref_name = 'images/' + ref_name
#                 #ref_name = ref_list[names_list.index(img_name)]
#                 mos = line_list[1].replace('\n', '')
#                 std = line_list[2].replace('\n', '')

#                 print(img_name, ref_name)
                    
#                 dist_img = prepare_image(Image.open(os.path.join('kadid10k', img_name)).convert("RGB")).to(device)
#                 ref_img = prepare_image(Image.open(os.path.join('kadid10k', ref_name)).convert("RGB")).to(device)
        
#                 ssim = ssim_matlab[step]
#                 gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
#                 mad = MAD_D(dist_img, ref_img, as_loss=False).item()
#                 vsi = vsi_matlab[step]
#                 lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
#                 nlpds = nlpd_matlab[step]
#                 wfile.write("{},{},{},{},{},{},{},{},{},{}\n".format(img_name.replace('\n', ''), ref_name.replace('\n', ''), \
#                                                                 mos, std, ssim, gmsd, mad, vsi, lpips, nlpds))

# def combine(temp):
#     with open('train_pair_{}.txt'.format(temp), 'w') as train_file:
#         with open('databaserelease2/splits2/1/live_train_score_kd.txt', 'r') as wfile:
#             lines = wfile.readlines()
#             ssim_list = []
#             gmsd_list = []
#             mad_list = []
#             vsi_list = []
#             lpips_list = []
#             nlpds_list = []
#             for line in lines:
#                 line_list = line.replace('\n', '').split(',')
#                 ssim_list.append(float(line_list[4]))
#                 gmsd_list.append(float(line_list[5]))
#                 mad_list.append(float(line_list[6]))
#                 vsi_list.append(float(line_list[7]))
#                 lpips_list.append(float(line_list[8]))
#                 nlpds_list.append(float(line_list[9]))
                
#             lens = len(lines)
#             idx_list = list(combinations(np.linspace(0,lens,lens, endpoint=False).astype(np.int16).tolist(), 2))
#             idx_list = random.sample(idx_list, 50000)
        
#             for step, (i,j) in enumerate(idx_list):
#                 pre_img = lines[i].replace('\n', '').split(',')
#                 lat_img = lines[j].replace('\n', '').split(',')
#                 img_ref_name1 = pre_img[:1]
#                 img_ref_name2 = lat_img[:1]

#                 gt_label = int(float(pre_img[2]) > float(lat_img[2]))

#                 exp = math.exp((float(pre_img[4]) - float(lat_img[4]))/abs(max(ssim_list) -min(ssim_list))/temp)
#                 ssim_label = exp/( 1.0 + exp)
#                 exp = math.exp(-(float(pre_img[5]) - float(lat_img[5]))/abs(max(gmsd_list) -min(gmsd_list))/temp)
#                 gmsd_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[6]) - float(lat_img[6]))/abs(max(mad_list) -min(mad_list))/temp)
#                 mad_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[7]) - float(lat_img[7]))/abs(max(vsi_list) -min(vsi_list))/temp)
#                 vsi_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[8]) - float(lat_img[8]))/abs(max(lpips_list) -min(lpips_list))/temp)
#                 lpips_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[9]) - float(lat_img[9]))/abs(max(nlpds_list) -min(nlpds_list))/temp)
#                 nlpds_label = exp/(1.0 + exp)
                
#                 train_file.write("{},{},{},{},{},{},{},{},{},{},{},1\n".format('databaserelease2/' + pre_img[0], \
#                                                                              'databaserelease2/' + pre_img[1], \
#                                                                              'databaserelease2/' + lat_img[0], \
#                                                                              'databaserelease2/' + lat_img[1], \
#                                                                              gt_label, ssim_label, gmsd_label, \
#                                                                              mad_label, vsi_label, lpips_label, nlpds_label))
                
#         with open('CSIQ/splits2/1/csiq_train_score_kd.txt', 'r') as wfile:
#             lines = wfile.readlines()
#             ssim_list = []
#             gmsd_list = []
#             mad_list = []
#             vsi_list = []
#             lpips_list = []
#             nlpds_list = []
#             for line in lines:
#                 line_list = line.replace('\n', '').split(',')
#                 ssim_list.append(float(line_list[4]))
#                 gmsd_list.append(float(line_list[5]))
#                 mad_list.append(float(line_list[6]))
#                 vsi_list.append(float(line_list[7]))
#                 lpips_list.append(float(line_list[8]))
#                 nlpds_list.append(float(line_list[9]))
                
#             lens = len(lines)
#             idx_list = list(combinations(np.linspace(0,lens,lens, endpoint=False).astype(np.int16).tolist(), 2))
#             idx_list = random.sample(idx_list, 50000)
        
#             for step, (i,j) in enumerate(idx_list):
#                 pre_img = lines[i].replace('\n', '').split(',')
#                 lat_img = lines[j].replace('\n', '').split(',')
#                 img_ref_name1 = pre_img[:1]
#                 img_ref_name2 = lat_img[:1]


#                 gt_label = int(float(pre_img[2]) > float(lat_img[2]))
#                 exp = math.exp((float(pre_img[4]) - float(lat_img[4]))/abs(max(ssim_list) -min(ssim_list))/temp)
#                 ssim_label = exp/( 1.0 + exp)
#                 exp = math.exp(-(float(pre_img[5]) - float(lat_img[5]))/abs(max(gmsd_list) -min(gmsd_list))/temp)
#                 gmsd_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[6]) - float(lat_img[6]))/abs(max(mad_list) -min(mad_list))/temp)
#                 mad_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[7]) - float(lat_img[7]))/abs(max(vsi_list) -min(vsi_list))/temp)
#                 vsi_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[8]) - float(lat_img[8]))/abs(max(lpips_list) -min(lpips_list))/temp)
#                 lpips_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[9]) - float(lat_img[9]))/abs(max(nlpds_list) -min(nlpds_list))/temp)
#                 nlpds_label = exp/(1.0 + exp)
                
#                 train_file.write("{},{},{},{},{},{},{},{},{},{},{},2\n".format('CSIQ/' + pre_img[0], 'CSIQ/' + pre_img[1], 'CSIQ/' + lat_img[0], 'CSIQ/' + lat_img[1], \
#                                                                             gt_label, ssim_label, gmsd_label, mad_label, vsi_label, lpips_label, nlpds_label))
                
#         with open('TID2013/splits2/1/tid_train_score_kd.txt', 'r') as wfile:
#             lines = wfile.readlines()
#             ssim_list = []
#             gmsd_list = []
#             mad_list = []
#             vsi_list = []
#             lpips_list = []
#             nlpds_list = []
#             for line in lines:
#                 line_list = line.replace('\n', '').split(',')
#                 ssim_list.append(float(line_list[4]))
#                 gmsd_list.append(float(line_list[5]))
#                 mad_list.append(float(line_list[6]))
#                 vsi_list.append(float(line_list[7]))
#                 lpips_list.append(float(line_list[8]))
#                 nlpds_list.append(float(line_list[9]))
                
#             lens = len(lines)
#             idx_list = list(combinations(np.linspace(0,lens,lens, endpoint=False).astype(np.int16).tolist(), 2))
#             idx_list = random.sample(idx_list, 100000)
        
#             for step, (i,j) in enumerate(idx_list):
#                 pre_img = lines[i].replace('\n', '').split(',')
#                 lat_img = lines[j].replace('\n', '').split(',')
#                 img_ref_name1 = pre_img[:1]
#                 img_ref_name2 = lat_img[:1]
                
#                 gt_label = int(float(pre_img[2]) > float(lat_img[2]))
#                 exp = math.exp((float(pre_img[4]) - float(lat_img[4]))/abs(max(ssim_list) -min(ssim_list))/temp)
#                 ssim_label = exp/( 1.0 + exp)
#                 exp = math.exp(-(float(pre_img[5]) - float(lat_img[5]))/abs(max(gmsd_list) -min(gmsd_list))/temp)
#                 gmsd_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[6]) - float(lat_img[6]))/abs(max(mad_list) -min(mad_list))/temp)
#                 mad_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[7]) - float(lat_img[7]))/abs(max(vsi_list) -min(vsi_list))/temp)
#                 vsi_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[8]) - float(lat_img[8]))/abs(max(lpips_list) -min(lpips_list))/temp)
#                 lpips_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[9]) - float(lat_img[9]))/abs(max(nlpds_list) -min(nlpds_list))/temp)
#                 nlpds_label = exp/(1.0 + exp)
                
#                 train_file.write("{},{},{},{},{},{},{},{},{},{},{},3\n".format('TID2013/'+pre_img[0], 'TID2013/'+pre_img[1], 'TID2013/'+lat_img[0], 'TID2013/'+lat_img[1], \
#                                                                             gt_label, ssim_label, gmsd_label, mad_label, vsi_label, lpips_label, nlpds_label))
        
#         with open('kadid10k/splits2/1/kadid10k_train_score_kd.txt', 'r') as wfile:
#             lines = wfile.readlines()
#             ssim_list = []
#             gmsd_list = []
#             mad_list = []
#             vsi_list = []
#             lpips_list = []
#             nlpds_list = []
#             for line in lines:
#                 line_list = line.replace('\n', '').split(',')
#                 ssim_list.append(float(line_list[4]))
#                 gmsd_list.append(float(line_list[5]))
#                 mad_list.append(float(line_list[6]))
#                 vsi_list.append(float(line_list[7]))
#                 lpips_list.append(float(line_list[8]))
#                 nlpds_list.append(float(line_list[9]))
                
#             lens = len(lines)
#             idx_list = list(combinations(np.linspace(0,lens,lens, endpoint=False).astype(np.int16).tolist(), 2))
#             idx_list = random.sample(idx_list, 200000)
        
#             for step, (i,j) in enumerate(idx_list):
#                 pre_img = lines[i].replace('\n', '').split(',')
#                 lat_img = lines[j].replace('\n', '').split(',')
#                 img_ref_name1 = pre_img[:1]
#                 img_ref_name2 = lat_img[:1]
                
#                 gt_label = int(float(pre_img[2]) > float(lat_img[2]))
#                 exp = math.exp((float(pre_img[4]) - float(lat_img[4]))/abs(max(ssim_list) -min(ssim_list))/temp)
#                 ssim_label = exp/( 1.0 + exp)
#                 exp = math.exp(-(float(pre_img[5]) - float(lat_img[5]))/abs(max(gmsd_list) -min(gmsd_list))/temp)
#                 gmsd_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[6]) - float(lat_img[6]))/abs(max(mad_list) -min(mad_list))/temp)
#                 mad_label = exp/(1.0 + exp)
#                 exp = math.exp((float(pre_img[7]) - float(lat_img[7]))/abs(max(vsi_list) -min(vsi_list))/temp)
#                 vsi_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[8]) - float(lat_img[8]))/abs(max(lpips_list) -min(lpips_list))/temp)
#                 lpips_label = exp/(1.0 + exp)
#                 exp = math.exp(-(float(pre_img[9]) - float(lat_img[9]))/(max(nlpds_list) -min(nlpds_list))/temp)
#                 nlpds_label = exp/(1.0 + exp)
                
#                 train_file.write("{},{},{},{},{},{},{},{},{},{},{},4.0\n".format('kadid10k/'+pre_img[0], 'kadid10k/'+pre_img[1], 'kadid10k/'+lat_img[0], 'kadid10k/'+lat_img[1], \
#                                                                             gt_label, ssim_label, gmsd_label, mad_label, vsi_label, lpips_label, nlpds_label))
    
# if __name__ == '__main__':
#     #srcc, plcc = test()
#     #print(str(srcc))
#     #print(str(plcc))
#     #train_pred()
#     combine(0.2)
# 	# test
# 	# dist_img = prepare_image(Image.open('I01.png').convert("RGB")).to(device)
# 	# ref_img = prepare_image(Image.open('I01_01_01.png').convert("RGB")).to(device)
# 	# ssim = SSIM_D(dist_img, ref_img, as_loss=False).item()
# 	# gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
# 	# mad = MAD_D(dist_img, ref_img, as_loss=False).item()
# 	# vsi = VSI_D(dist_img, ref_img, as_loss=False).item()
# 	# lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
# 	# nlpds = VIF_D(dist_img, ref_img, as_loss=False).item()
# 	# print(ssim, gmsd, mad, vsi, lpips, nlpds)

# 	# dist_img = prepare_image(Image.open('I01.png').convert("RGB")).to(device)
# 	# ref_img = prepare_image(Image.open('I01_01_02.png').convert("RGB")).to(device)
# 	# ssim = SSIM_D(dist_img, ref_img, as_loss=False).item()
# 	# gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
# 	# mad = MAD_D(dist_img, ref_img, as_loss=False).item()
# 	# vsi = VSI_D(dist_img, ref_img, as_loss=False).item()
# 	# lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
# 	# nlpds = VIF_D(dist_img, ref_img, as_loss=False).item()
# 	# print(ssim, gmsd, mad, vsi, lpips, nlpds)

# 	# dist_img = prepare_image(Image.open('I01.png').convert("RGB")).to(device)
# 	# ref_img = prepare_image(Image.open('I01_01_03.png').convert("RGB")).to(device)
# 	# ssim = SSIM_D(dist_img, ref_img, as_loss=False).item()
# 	# gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
# 	# mad = MAD_D(dist_img, ref_img, as_loss=False).item()
# 	# vsi = VSI_D(dist_img, ref_img, as_loss=False).item()
# 	# lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
# 	# nlpds = VIF_D(dist_img, ref_img, as_loss=False).item()
# 	# print(ssim, gmsd, mad, vsi, lpips, nlpds)

# 	# dist_img = prepare_image(Image.open('I01.png').convert("RGB")).to(device)
# 	# ref_img = prepare_image(Image.open('I01_01_04.png').convert("RGB")).to(device)
# 	# ssim = SSIM_D(dist_img, ref_img, as_loss=False).item()
# 	# gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
# 	# mad = MAD_D(dist_img, ref_img, as_loss=False).item()
# 	# vsi = VSI_D(dist_img, ref_img, as_loss=False).item()
# 	# lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
# 	# nlpds = VIF_D(dist_img, ref_img, as_loss=False).item()
# 	# print(ssim, gmsd, mad, vsi, lpips, nlpds)

# 	# dist_img = prepare_image(Image.open('I01.png').convert("RGB")).to(device)
# 	# ref_img = prepare_image(Image.open('I01_01_05.png').convert("RGB")).to(device)
# 	# ssim = SSIM_D(dist_img, ref_img, as_loss=False).item()
# 	# gmsd = GMSD_D(dist_img, ref_img, as_loss=False).item()
# 	# mad = MAD_D(dist_img, ref_img, as_loss=False).item()
# 	# vsi = VSI_D(dist_img, ref_img, as_loss=False).item()
# 	# lpips = LPIPS_D(dist_img, ref_img, as_loss=False).item()
# 	# nlpds = VIF_D(dist_img, ref_img, as_loss=False).item()
# 	# print(ssim, gmsd, mad, vsi, lpips, nlpds)
