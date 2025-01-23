import sys

sys.path.append('../')
import torch.backends.cudnn as cudnn
import torchvision.models as models
from core.builder import setup_env

import core.config as config
from runner.utilsq import Adaptive_BN, get_train_samples, convert_to_QuantSuperModel

from data import imagenet_dali
from logger.meter import *
from runner.evaluatorq import evaluation_quant_model_2to8, evaluation_model, evaluation_model_using_AdaptiveBN
import numpy as np
import random
from core.config import cfg, load_configs
from moduleq.mobilenetv2 import mobilenet_v2q

config.load_configs()
logger = logging.get_logger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

setup_seed(20230101)

def main():
    setup_env()
    # model = models.__dict__[cfg.ARCH](pretrained=True)
    model = mobilenet_v2q(pretrained=False)
    print(model)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPUS[0])
        model = model.cuda()
        cudnn.benchmark = True

    train_loader, _ = imagenet_dali.get_imagenet_iter_torch('train', cfg.DATASET.data_path,
                                                            cfg.DATASET.train_batch_size,
                                                            num_threads=2, crop=224, device_id=cfg.GPUS[0])

    test_loader, _ = imagenet_dali.get_imagenet_iter_torch('val', cfg.DATASET.data_path, cfg.DATASET.eval_batch_size,
                                                           num_threads=2, crop=224, device_id=cfg.GPUS[0])

    wq_params = {'n_bits': cfg.w_bit, 'scale_method': 'mse', 'leaf_param': True}
    aq_params = {'n_bits': cfg.a_bit, 'scale_method': 'mse', 'leaf_param': True}
    search_space = {
        'w_bit_list': cfg.SEARCH_SPACE.w_bit_list,
        'a_bit_list': cfg.SEARCH_SPACE.a_bit_list,
        'w_sym_list': cfg.SEARCH_SPACE.w_sym_list,
        'a_sym_list': cfg.SEARCH_SPACE.a_sym_list,
        'channel_wise_list': cfg.SEARCH_SPACE.channel_wise_list,
    }
    qnn = convert_to_QuantSuperModel(model, wq_params=wq_params, aq_params=aq_params, quantizer=cfg.quantizer,
                                     search_space=search_space)

    if not cfg.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(train_loader=train_loader, num_samples=cfg.num_samples)
    qnn.set_quant_state(True, True)
    with torch.no_grad():
        _ = qnn(cali_data.cuda())

    print('load model...')
    ckpt = torch.load(cfg.super_model, map_location='cuda')
    qnn.load_state_dict(ckpt['state_dict'])

    wqc1 = []
    wqc3 = []
    wq = []
 
    for pname, p in qnn.named_parameters():
        # if 'conv2d' in pname:
        # print(pname,p.size())
        if 'wq' in pname:
            logger.info('weight_param:{}, {}'.format(pname,p.size()))
            if p.ndimension() == 4:
                H,W,K1,K2 = p.shape
                if W*K1*K2!=27:
                    reshap_p = p.view(H,-1)
                    wq.append([pname,[H,W,K1,K2],reshap_p.detach()]) 
            # elif p.ndimension() == 2:
            #     H,W = p.shape
            #     wq.append([pname,[H,W],p.detach()])                 

    N = cfg.w_bit
    types = 2 ** N    # (4,8,16)
    wise = 1
    Maxdis = [0]*types*wise

    # search_space = {
    #     # 'w_bit_list': [2, 3, 4, 5, 6, 7, 8],
    #     # 'a_bit_list': [2, 3, 4, 5, 6, 7, 8],
    #     'w_bit_list': [4],
    #     'a_bit_list': [4],        
    #     'w_sym_list': [True, False],
    #     'a_sym_list': [True, False],
    #     'channel_wise_list': [True, False],
    # }
    # evaluation_quant_model_2to8(model=qnn, train_loader=train_loader, test_loader=test_loader,
    #                             search_space=search_space)
    for [pname,p_shape,p] in wq:
        H,W = p.shape
        logger.info("{} : {}, min:{}, max:{}".format(pname,p.size(),torch.min(p), torch.max(p)))
        
        # for i in range(int(H/wise)):# in
        for i in range(int(W/wise)): # out
            # subp = p[j,i*wise+s:i*wise+s+1,:]
            Dis1=[0]*types
            # for l in subp:
            #     for data in l:
            #         Dis1[int(data)]+=1
            # for k in range(len(Dis1)):
            #     # print(k,i,k+s*types)
            #     if Dis1[k]>Maxdis[k+s*types]:
            #         Maxdis[k+s*types]=Dis1[k]
            # subp = p[i*wise:(i+1)*wise,:]  # in
            subp = p[:,i*wise:(i+1)*wise]   # out
            if i ==0:
                logger.info("subp_size:{}".format(subp.size()))
            for l in subp:
                for data in l:
                    Dis1[int(data)+2 ** (N-1)]+=1
            for k in range(len(Dis1)):
                if Dis1[k]>Maxdis[k]:
                    Maxdis[k]=Dis1[k]
    logger.info("Max w from {} to {} is: {}, sum: {}".format(-1* (2 ** (N-1)), 2 ** (N-1)-1, Maxdis,sum(Maxdis)))
    # model_state_dict = qnn.state_dict()
    # state = {
    #     'state_dict': model_state_dict
    # }
    # torch.save(state, '/home/ming/coding/EQ-Net-main/mobilenet_v2_wq.pt')

if __name__ == '__main__':
    main()
