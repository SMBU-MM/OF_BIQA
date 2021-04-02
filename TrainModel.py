import os
import time
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from resnet18 import resnet18 as netF
from resnet18 import net_quality as netQ
from resnet18 import net_domain as netD
from resnet18 import net_annotators as netA

from ImageDataset import ImageDataset
from Transformers import AdaptiveResize
from tensorboardX import SummaryWriter
import prettytable as pt
import numpy as np
from utils import Binary_Loss, FocalBCELoss, VAT
from scipy.optimize import curve_fit

torch.cuda.empty_cache() 

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.count = 0
        self.loss_count = 0
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.config.seed)
        
        self.train_transform = transforms.Compose([
            #transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        
        self.train_batch_size = self.config.batch_size
        self.test_batch_size = 1
        
        train_txt = 'train_pair_koniq.txt'
        self.train_batch_size = config.batch_size
        self.test_batch_size = 1

        self.train_data = ImageDataset(csv_file=os.path.join(config.trainset, train_txt),
                                       img_dir=config.trainset,
                                       transform=self.train_transform,
                                       test=False)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=32)

        # testing set configuration
        self.live_data = ImageDataset(csv_file=os.path.join(config.live_set, 'splits2', str(config.split), 'live_test_779.txt'),
                                      img_dir=config.live_set,
                                      transform=self.test_transform,
                                      test=True)

        self.live_loader = DataLoader(self.live_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=8)

        self.csiq_data = ImageDataset(csv_file=os.path.join(config.csiq_set, 'splits2', str(config.split), 'csiq_test_886.txt'),
                                      img_dir=config.csiq_set,
                                      transform=self.test_transform,
                                      test=True)

        self.csiq_loader = DataLoader(self.csiq_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=8)

        self.tid2013_data = ImageDataset(csv_file=os.path.join(config.tid2013_set, 'splits2', str(config.split), 'tid_test_3000.txt'),
                                         img_dir=config.tid2013_set,
                                         transform=self.test_transform,
                                         test=True)

        self.tid2013_loader = DataLoader(self.tid2013_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=8)

        self.kadid10k_data = ImageDataset(csv_file=os.path.join(config.kadid10k_set, 'splits2', str(config.split), 'kadid10k_test_10125.txt'),
                                         img_dir=config.kadid10k_set,
                                         transform=self.test_transform,
                                         test=True)

        self.kadid10k_loader = DataLoader(self.kadid10k_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=8)

        self.koniq_data = ImageDataset(csv_file=os.path.join(config.koniq_set, 'splits2', str(config.split), 'koniq10k_test_10073.txt'),
                                         img_dir=config.koniq_set,
                                         transform=self.test_transform,
                                         test=True)

        self.koniq_loader = DataLoader(self.koniq_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=8)

        self.spaq_data = ImageDataset(csv_file=os.path.join(config.spaq_set,'spaq_test_11125.txt'),
                                         img_dir=config.spaq_set,
                                         transform=self.test_transform,
                                         test=True)

        self.spaq_loader = DataLoader(self.spaq_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=8)
        
        # mapping loader
        self.mapping_data = ImageDataset(csv_file=os.path.join(config.live_set, 'splits2', str(config.split), 'live_mapping.txt'),
                                      img_dir=config.live_set,
                                      transform=self.test_transform,
                                      test=True)

        self.mapping_loader = DataLoader(self.mapping_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=8)

        self.writer = SummaryWriter(self.config.runs_path)
        self.netF = nn.DataParallel(netF(pretrained=False).cuda())
        self.netF.load_state_dict(torch.load('MNL_DA-00008.pt')['netF_dict'])
        # fix the param of netF
        if self.config.fc == True:
            for name, para in self.netF.named_parameters():
                if 'fc.' in name:
                    para.requires_grad=True
                else:
                    para.requires_grad=False
        else:
            for name, para in self.netF.named_parameters():
                para.requires_grad=True
        
        self.netQ = nn.DataParallel(netQ().cuda())
        self.netQ.load_state_dict(torch.load('MNL_DA-00008.pt')['netQ_dict'])

        self.netD = nn.DataParallel(netD()).cuda()
        self.netA = nn.DataParallel(netA()).cuda()
        # loss function
        self.bin_fn = Binary_Loss().cuda()
        self.bce_fn = FocalBCELoss(gamma=self.config.gamma).cuda()
        print(self.netF, self.netQ, self.netD, self.netA)
        self.save_prefix = "MNL_DA"
        # oracle's log variance
        self.sensitivity = torch.load('MNL_DA-00008.pt')['sensitivity']
        self.specificity = torch.load('MNL_DA-00008.pt')['specificity']
        self.initial_lr = self.config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr
        # for domain adaptation
        self.optimizer = optim.Adam([{'params': self.netF.parameters(), 'lr': lr},
                                     {'params': self.netQ.parameters(), 'lr': lr},
                                     {'params': self.netD.parameters(), 'lr': lr},
                                     {'params': self.netA.parameters(), 'lr': lr},
                                     {'params': self.sensitivity, 'lr': self.config.ss_lr},
                                     {'params': self.specificity, 'lr': self.config.ss_lr}], lr=lr, weight_decay=5e-4)
      
        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results_srcc = {'live': [], 'csiq': [], 'tid2013': [], 'kadid': [], 'spaq': [], 'koniq': []}
        self.test_results_plcc = {'live': [], 'csiq': [], 'tid2013': [], 'kadid': [], 'spaq': [], 'koniq': []}
        
        self.ckpt_path = self.config.ckpt_path
        self.max_epochs = self.config.max_epochs
        self.epochs_per_eval = self.config.epochs_per_eval
        self.epochs_per_save = self.config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            print('**********************************************************************************')
            print("ckpt:", ckpt)
            print('start from the pretrained model of Save Model')
            print('**********************************************************************************')
            self._load_checkpoint(ckpt=ckpt)
            
        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=self.config.decay_interval,
                                             gamma=self.config.decay_ratio)

    def fit(self):        
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        
        for name, para in self.netF.named_parameters():
            print('{} parameters requires_grad:{}'.format(name, para.requires_grad))

        running_loss_q = 0 if epoch == 0 else self.train_loss[-1][0]
        running_loss_d = 0 if epoch == 0 else self.train_loss[-1][1]
        running_loss_c = 0 if epoch == 0 else self.train_loss[-1][2]
        running_loss_m = 0 if epoch == 0 else self.train_loss[-1][3]
        running_loss = 0 if epoch == 0 else self.train_loss[-1][4]
        
        loss_d_corrected = 0.0
        loss_q_corrected = 0.0
        loss_c_corrected = 0.0
        loss_m_corrected = 0.0
        running_duration = 0.0
        self.netF.train()
        self.netQ.train()
        self.netD.train()
        self.netA.train()

        start_steps = epoch * len(self.train_loader)
        total_steps = self.config.max_epochs * len(self.train_loader)

        # start training
        for step, sample_batched in enumerate(self.train_loader, 0):
            if step < self.start_step:
                continue
            #x1, x2 are the synthetic images, x3 is the real distortion image
            x1, x2, x3, x4, y_label, vot = sample_batched['I1'], sample_batched['I2'], sample_batched['I3'], \
                           sample_batched['I4'], sample_batched['y'], sample_batched['v']
           
            x1 = Variable(x1).cuda()
            x2 = Variable(x2).cuda()
            x3 = Variable(x3).cuda()
            x4 = Variable(x4).cuda()

            y_label = Variable(y_label).cuda()
            vot =  Variable(vot).cuda()
            # zero_grad
            self.optimizer.zero_grad()
            feats_x1 = self.netF(x1)
            feats_x2 = self.netF(x2)
            feats_x3 = self.netF(x3)
            feats_x4 = self.netF(x4)

            ############################################################################################
            # quality prediction loss
            ############################################################################################
            y1, y1_var, _, _ = self.netQ(feats_x1)
            y2, y2_var, _, _ = self.netQ(feats_x2)
            # the loss of quality prediction
            y_diff = y1 - y2
            # var = torch.ones(y1.shape[0], y1.shape[1]).cuda()
            y_var = y1_var**2 + y2_var**2 + 1e-4
            p = 0.5 * (1 + torch.erf(y_diff/torch.sqrt(2*y_var)))
            self.loss_q, _= self.bin_fn(p, y_label, torch.sigmoid(self.sensitivity), torch.sigmoid(self.specificity))
            #############################################################################################
            # classification loss
            #############################################################################################
            c1 = self.netA(torch.cat([feats_x1, feats_x2], dim=1))
            self.loss_c = self.bce_fn(c1.view(-1), vot.view(-1).to(torch.float32).detach())
            #############################################################################################
            # Domain discriminator loss
            #############################################################################################
            # here is substract because we are not add the gneralized reverse layer (GRL)
            # x1,x2 from source, x3 from target
            feats_combined = torch.cat([feats_x1, feats_x2, feats_x3, feats_x4], dim=0)
            #y_combined = torch.cat([y1, y2, y3], dim=0)
            # assign source domain with label 0 .type(torch.LongTensor)
            D_src = torch.zeros(x1.shape[0]).cuda() # Discriminator Label to real
            # assign target domain with label 1 （as same as DANN paper）
            D_tgt = torch.ones(x3.shape[0]).cuda() # Discriminator Label to fake
            domain_labels = torch.cat([D_src, D_src, D_tgt, D_tgt], dim=0).cuda()
            tp = float(step + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * tp)) - 1
            domain_preds = self.netD(feats_combined, alpha)
            self.loss_d = self.bce_fn(domain_preds.view(-1), domain_labels.view(-1))
            #############################################################################################
            # inter-domain pixel-level mixup 
            #############################################################################################
            mix_ratio = np.random.beta(self.config.beta, self.config.beta)
            mix_ratio = round(mix_ratio, 2)
            if (mix_ratio >= 0.5 and mix_ratio < (0.5 + self.config.clip_thr)):
                mix_ratio = 0.5 + self.config.clip_thr
            if (mix_ratio > (0.5 - self.config.clip_thr) and mix_ratio < 0.5):
                mix_ratio = 0.5 - self.config.clip_thr
            label_mix = mix_ratio * torch.cat([D_src, D_src], dim=0) + (1.0-mix_ratio) *  torch.cat([D_tgt, D_tgt], dim=0)
            img_mix =   mix_ratio * torch.cat([x1, x2], dim=0) + (1.0 - mix_ratio) * torch.cat([x3, x4], dim=0)
            label_pred_mix = self.netD(self.netF(img_mix), alpha)
            self.loss_m  = self.bce_fn(label_pred_mix.view(-1), label_mix.view(-1))
            #############################################################################################
            # total loss
            #############################################################################################
            self.loss = self.loss_q + self.config.weight_d*self.loss_d + self.config.weight_c*self.loss_c + self.config.weight_m*self.loss_m
            self.loss.backward()
            # updata quality nets first and then update feature nets
            self.optimizer.step()
            # statistics
            running_loss_q = beta * running_loss_q + (1 - beta) * self.loss_q.data.item()
            loss_q_corrected = running_loss_q / (1 - beta ** local_counter)

            running_loss_d = beta * running_loss_d + (1 - beta) * self.loss_d.data.item()
            loss_d_corrected = running_loss_d / (1 - beta ** local_counter)

            running_loss_c = beta * running_loss_c + (1 - beta) * self.loss_c.data.item()
            loss_c_corrected = running_loss_c / (1 - beta ** local_counter)

            running_loss_m = beta * running_loss_m + (1 - beta) * self.loss_m.data.item()
            loss_m_corrected = running_loss_m / (1 - beta ** local_counter)

            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            self.loss_count += 1
            if self.loss_count % 100 == 0:
                self.writer.add_scalars('data/Corrected_Loss', {'loss_d': loss_d_corrected,
                                                      'loss_q': loss_q_corrected,
                                                      'loss_c': loss_c_corrected,
                                                      'loss': loss_corrected
                                                      }, self.loss_count)
                self.writer.add_scalars('data/Uncorrected_Loss', {'loss_d': self.loss_d.data.item(),
                                                      'loss_q': self.loss_q.data.item(),
                                                      'loss_c': self.loss_c.data.item(),
                                                      'loss': self.loss.data.item()
                                                      }, self.loss_count)
            
            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d/%d) [Loss_q = %.4f, Loss_d = %.4f, Loss_c = %.4f, Loss_m = %.4f, Loss = %.4f] alpha=%.04f (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, len(self.train_loader), loss_q_corrected, loss_d_corrected, loss_c_corrected, \
                                loss_m_corrected, loss_corrected, alpha,  examples_per_sec, duration_corrected))
            local_counter += 1
            self.start_step = 0
            start_time = time.time()
            
        self.train_loss.append([loss_q_corrected, loss_d_corrected, loss_c_corrected, loss_m_corrected, loss_corrected])
        self.scheduler.step()
        # print SRCC each epoch
        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            srcc, plcc = self.eval()
            self.writer.add_scalars('data/SRCC', {"LIVE": srcc['live'], "CSIQ": srcc['csiq'],
                                                 "TID2013": srcc['tid2013'], "Kadid": srcc['kadid'],
                                                 "Spaq": srcc['spaq'], "Koniq":srcc['koniq']
                                                 }, epoch+1)
            self.writer.add_scalars('data/PLCC', {"LIVE": plcc['live'], "CSIQ": plcc['csiq'],
                                                 "TID2013": plcc['tid2013'], "Kadid": plcc['kadid'],
                                                 "Spaq": plcc['spaq'], "Koniq":plcc['koniq']
                                                 }, epoch+1)

            self.test_results_srcc['live'].append(srcc['live'])
            self.test_results_srcc['csiq'].append(srcc['csiq'])
            self.test_results_srcc['tid2013'].append(srcc['tid2013'])
            self.test_results_srcc['kadid'].append(srcc['kadid'])
            self.test_results_srcc['spaq'].append(srcc['spaq'])
            self.test_results_srcc['koniq'].append(srcc['koniq'])
            
            self.test_results_plcc['live'].append(plcc['live'])
            self.test_results_plcc['csiq'].append(plcc['csiq'])
            self.test_results_plcc['tid2013'].append(plcc['tid2013'])
            self.test_results_plcc['kadid'].append(plcc['kadid'])
            self.test_results_plcc['spaq'].append(plcc['spaq'])
            self.test_results_plcc['koniq'].append(plcc['koniq'])
            
            tb = pt.PrettyTable()
            tb.field_names = ["SRCC", "LIVE", "CSIQ", "TID2013", "KADID10K", "SPAQ", "KONIQ"]
            tb.add_row(['Ours', srcc['live'], srcc['csiq'], srcc['tid2013'], \
                            srcc['kadid'], srcc['spaq'], srcc['koniq']])
            tb.add_row(["PLCC", "LIVE", "CSIQ", "TID2013", "KADID10K", "SPAQ", "KONIQ"])
           
            tb.add_row(['Ours', plcc['live'], plcc['csiq'], plcc['tid2013'], \
                        plcc['kadid'], plcc['spaq'], plcc['koniq']])
            print(tb)
            f = open(self.config.result_path + r'results_{}.txt'.format(epoch), 'w')
            f.write(str(tb))
            f.close()
            
        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.save_prefix, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
           
            self._save_checkpoint({
                'epoch': epoch,
                'netF_dict': self.netF.state_dict(),
                'netQ_dict': self.netQ.state_dict(),
                'netD_dict': self.netD.state_dict(),
                'netA_dict': self.netA.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'sensitivity': self.sensitivity,
                'specificity': self.specificity,
                'train_loss': self.train_loss,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)
        return 0

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.sensitivity = checkpoint['sensitivity']
            self.specificity = checkpoint['specificity']
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.test_results = checkpoint['test_results_srcc']
            self.test_results = checkpoint['test_results_plcc']
            
            self.netF.load_state_dict(checkpoint['netF_dict'])
            self.netQ.load_state_dict(checkpoint['netQ_dict'])
            self.netD.load_state_dict(checkpoint['netD_dict'])
            self.netA.load_state_dict(checkpoint['netA_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))
    
    def eval_single(self, netF, netQ, dataloader, popt=None):
        q_mos = []
        q_bar = []
        # q_map = []
        for step, sample_batched in enumerate(dataloader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x).cuda()
            feats = netF(x)
            y_bar,_,_,_ = netQ(feats)
            y_bar = y_bar.cpu().item()
            q_mos.append(y.item())
            q_bar.append(y_bar)        
        srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_bar)[0]
        plcc = scipy.stats.mstats.pearsonr(x=q_mos, y=q_bar)[0]
        return srcc, plcc
            
    def eval(self):
        srcc = {}
        plcc = {}
        self.netF.eval()
        self.netQ.eval()
        #self.netD.eval()
        #popt = self._mapping()
        srcc['live'], plcc['live'] = self.eval_single(self.netF, self.netQ, self.live_loader)
        srcc['csiq'], plcc['csiq'] = self.eval_single(self.netF, self.netQ, self.csiq_loader)
        srcc['tid2013'], plcc['tid2013'] = self.eval_single(self.netF, self.netQ, self.tid2013_loader)
        srcc['kadid'], plcc['kadid'] = self.eval_single(self.netF, self.netQ, self.kadid10k_loader)
        srcc['spaq'], plcc['spaq'] = self.eval_single(self.netF, self.netQ, self.spaq_loader)
        srcc['koniq'], plcc['koniq'] = self.eval_single(self.netF, self.netQ, self.koniq_loader)
        return srcc, plcc

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

