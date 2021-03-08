import argparse
import TrainModel as TrainModel
import os
from shutil import copyfile

cuda_idx = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_idx

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--clip_thr", type=float, default=0.3)

    parser.add_argument("--weight_d", type=float, default=0.08)
    parser.add_argument("--weight_c", type=float, default=0.08)
    parser.add_argument("--weight_m", type=float, default=0.02)

    parser.add_argument("--voting", type=bool, default=False)
    parser.add_argument("--pretrained", type=bool, default=False)
    
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default=r"../gen/")
    parser.add_argument("--live_set", type=str, default=r"../IQA_database/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default=r"../IQA_database/CSIQ/")
    parser.add_argument("--tid2013_set", type=str, default=r"../IQA_database/TID2013/")
    parser.add_argument("--kadid10k_set", type=str, default=r"../IQA_database/kadid10k/")
    parser.add_argument("--koniq_set", type=str, default=r"../IQA_database/koniq-10k/")
    parser.add_argument("--spaq_set", type=str, default=r"../IQA_database/SPAQ/")

    checkpoint_path = r"./checkpoints_d0.08_g0.08_m0.04_interdomain/"
    parser.add_argument('--ckpt', default=checkpoint_path, type=str, help='name of the checkpoint to load')
    parser.add_argument('--ckpt_path', default= checkpoint_path +'checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--result_path', default= checkpoint_path + 'results/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--p_path', default= checkpoint_path + 'p/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--runs_path', default= checkpoint_path + 'runs/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--codes', default= checkpoint_path + 'codes/', type=str,
                        metavar='PATH', help='path to checkpoints')
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--ss_lr", type=float, default=3e-7)
    parser.add_argument("--fc", type=bool, default=False)
    
    parser.add_argument("--decay_interval", type=int, default=2)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    return parser.parse_args()

def main(cfg):
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        t.eval()
        

if __name__ == "__main__":
    config = parse_config()
    print('config.weight_d', config.weight_d)
    print('config.weight_c', config.weight_c)
    print('config.lr', config.lr)
    print('cuda_idx', cuda_idx)
    print('gamma', config.gamma)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.p_path):
        os.makedirs(config.p_path)
    if not os.path.exists(config.codes):
        os.makedirs(config.codes)
    # copyfiles
    copyfile('Main.py', os.path.join(config.codes, 'Main.py'))
    copyfile('ImageDataset.py', os.path.join(config.codes, 'ImageDataset.py'))
    copyfile('resnet18.py', os.path.join(config.codes, 'resnet18.py'))
    copyfile('TrainModel.py', os.path.join(config.codes, 'TrainModel.py'))
    copyfile('Transformers.py', os.path.join(config.codes, 'Transformers.py'))
    copyfile('utils.py', os.path.join(config.codes, 'utils.py'))

    with open(os.path.join(config.ckpt, 'config.txt'), 'w') as wfile:
        wfile.write('gpu_idx:{}\n'.format(cuda_idx))
        wfile.write('config.weight_d:{}\n'.format(config.weight_d))
        wfile.write('config.weight_c:{}\n'.format(config.weight_c))
        wfile.write('config.lr:{}\n'.format(config.lr))
        wfile.write('config.ss_lr:{}\n'.format(config.ss_lr))
        wfile.write('config.voting:{}\n'.format(config.voting))
        wfile.write('config.decay_interval:{}\n'.format(config.decay_interval))

        wfile.write('config.gamma for focal loss:{}\n'.format(config.gamma))

        wfile.write('set weight_c = 0.0 after 1 epoch')
    main(config)
