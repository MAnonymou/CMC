import sys

sys.path.append('../')
import torch.backends.cudnn as cudnn
import torchvision.models as models
from core.builder import setup_env
import os
import core.config as config
from runner.utilsq import Adaptive_BN, get_train_samples, convert_to_QuantSuperModel

from data import imagenet_dali
from logger.meter import *
from runner.evaluatorq import evaluation_quant_model_2to8, evaluation_model, evaluation_model_using_AdaptiveBN
import numpy as np
import random
from core.config import cfg, load_configs
from moduleq.mobilenetv2 import mobilenet_v2q
import math

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


def generate_sic(output_dir,bits,wlist):
    verilog='''module SIC
#(
parameter sic_n = {},
parameter data_width_in = {},
parameter data_width_out = {}
)
(
'''.format(sum(wlist),bits,2*bits-1)
    r = 2 ** (bits-1)
    for i in range(len(wlist)):
        i-= r
        if i<0:
            verilog+='''\tinput  [{}*data_width_in-1:0] Data_in_n{},\n'''.format(wlist[i+r],abs(i))
        elif i>0:
            verilog+='''\tinput  [{}*data_width_in-1:0] Data_in_p{},\n'''.format(wlist[i+r],i)

    verilog+='''\toutput [sic_n*data_width_out-1:0] Data_out
);
'''
    accumulate_wlist = []
    accu=0
    for amt in wlist:
        accu+=amt
        accumulate_wlist.append(accu)

    # verilog+="parameter nn4={}, nn3={}, nn2={}, nn1={}, np1={}, np2={}, np3={};\n".format(accumulate_wlist[0],accumulate_wlist[1],accumulate_wlist[2],accumulate_wlist[3],accumulate_wlist[5],accumulate_wlist[6],accumulate_wlist[7])
    verilog+='''parameter '''
    for i in range(len(wlist)):
        i-=r
        if i<0:
            verilog+='''nn{}= {},'''.format(abs(i),accumulate_wlist[i+r])
        elif i>0 and (i+r+1) != len(wlist):
            verilog+='''np{}= {},'''.format(abs(i),accumulate_wlist[i+r])
        elif (i+r+1) == len(wlist):
            verilog+='''np{}= {};\n'''.format(abs(i),accumulate_wlist[i+r])

    verilog+="genvar i;\n"
    for i in range(len(wlist)):
        i-=r
        if i==-1*r:
            verilog+='''
generate
    for (i=0;i<nn{};i=i+1) begin
        wn{}  wn{}_u (.din(Data_in_n{}[data_width_in*(i+1)-1 -: data_width_in]),.dout(Data_out[data_width_out*(i+1)-1 -: data_width_out]));
    end
endgenerate
'''.format(abs(i),abs(i),abs(i),abs(i))
        elif i>-1*r and i!=0:
            verilog+='''
generate
    for (i=nn{};i<nn{};i=i+1) begin
        wn{}  wn{}_u (.din(Data_in_n{}[data_width_in*(i+1-nn{})-1 -: data_width_in]),.dout(Data_out[data_width_out*(i+1)-1 -: data_width_out]));
    end
endgenerate
'''.format(abs(i-1),abs(i),abs(i),abs(i),abs(i),abs(i-1))
        elif i == 1:
            verilog+='''
generate
    for (i=nn{};i<np{};i=i+1) begin
        assign Data_out[data_width_out*(i+1)-1 -: data_width_out] = {{Data_in_p{}[data_width_in*(i+1-nn{})-1 -: data_width_in],1'b0}};
    end
endgenerate
'''.format(i,i,i,i)
        else:
            verilog+='''
generate
    for (i=np{};i<np{};i=i+1) begin
        wp{}  wp{}_u (.din(Data_in_p{}[data_width_in*(i+1-np{})-1 -: data_width_in]),.dout(Data_out[data_width_out*(i+1)-1 -: data_width_out]));
    end
endgenerate
'''.format(i-1,i,i,i,i,i-1)
    verilog+="endmodule"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    with open(output_dir+"/SIC.v","w") as f:
        f.write(verilog)

def generate_mmwire(x,bits,output_dir,wlist):
    xname = x[0].split('.')

    if len(xname)== 7 or len(xname) == 8:
        blocknum = xname[2]
        layername = 'b'+xname[2]+'c'+xname[4]+'s'+xname[len(xname)-1]
    elif len(xname)== 6:
        blocknum = xname[2]
        layername = 'b'+xname[2]+'c0s'+xname[len(xname)-1]
    elif len(xname)== 5:
        blocknum = '19'
        layername = 'b19c0s'+xname[len(xname)-1]
    else:
        print('error layer is: {}'.format(x[0]))

    wq = x[1]
    w, h = wq.shape #input wq is w^t
    r = 2 ** (bits-1)
    b = 2 ** (4-bits)
    verilog='''module {}_wire
#(
parameter array_m = {},
parameter array_n = {},
parameter data_width = {}
)
(
    input  [array_m*data_width-1:0]    Data_in,
'''.format(layername,w,h,bits)

    usedw1 = [0]*(len(wlist))
    # verilog+="always @(*) begin\n"
    # verilog+="case(en) \n 1'b1: begin\n"
    for j in range(h):
        for i in range(w):
            for k in range(len(wlist)):
                k-=r
                if int(torch.floor(wq[i][j]/b)) == k:
                    usedw1[k+r]+=1
    # for wx in usedw1:
    #     wx -= 1    
    First = True
    for i in range(len(wlist)):
        if usedw1[i]>0:
            i-=r
            if i != 0:
                if i<0:
                    sign = 'n'
                else:
                    sign = 'p'

                if First:
                    verilog+='''\toutput  [{}*data_width-1:0] Data_out_{}{}'''.format(usedw1[i+r],sign,abs(i))
                    First = False
                else:
                    verilog+=''',\n\toutput  [{}*data_width-1:0] Data_out_{}{}'''.format(usedw1[i+r],sign,abs(i))
                    First = False
         
    verilog+='''\n);\n'''
    usedw = [0]*(len(wlist)+2)

    # verilog+="always @(*) begin\n"
    # verilog+="case(en) \n 1'b1: begin\n"
    for j in range(h):
        for i in range(w):
            for k in range(len(wlist)):
                k-=r
                if int(torch.floor(wq[i][j]/b)) == k:
                    if int(torch.floor(wq[i][j]/b))!=0:
                        if int(torch.floor(wq[i][j]/b))>0:
                            sign = 'p'
                        else:
                            sign = 'n'
                        verilog+="assign Data_out_{}{}[data_width*{}-1 -:data_width] = Data_in[data_width*{}-1 -:data_width];\n".format(sign,abs(k),usedw[k+r]+1,i+1)
                        usedw[k+r]+=1                        

    # verilog+="\tend\ndefault: Data_out = 0;\nendcase\nend\n"
    verilog+="endmodule"
    # print(x[0],"wire_in",sum(usedw))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+"/{}_wire.v".format(layername),"w") as f:
        f.write(verilog)
    # for wx in usedw:
    #     wx -= 1
    usedw[-1]=layername
    usedw[-2]=w
    return usedw

def generate_top_half(x,bits,wlist,output_dir,blocks,shapelist,usedwqc):
    wq = x[1]
    w, h = wq.shape #input wq is w^t    
    H = max(w,h)
    subsmax = 0
    r = 2 ** (bits-1)
    b = 2 ** (4-bits)
    for [name,shape] in shapelist:
        Wid = shape[1]
        if Wid> subsmax:
            subsmax = Wid
    
    maxlist = []
    input_num_list=[]
    # for list in usedwqc:
    for i in range(len(usedwqc)-1):
        i+=1
        list = usedwqc[i-1]
        name1 = list[-1].split('b')
        name2 = name1[1].split('c')
        blk = int(name2[0])
        name3 = name2[1].split('s')
        layer = int(name3[0])
        sub = int(name3[1])  

        list0 = usedwqc[i]    
        name10 = list0[-1].split('b')
        name20 = name10[1].split('c')
        blk0 = int(name20[0])
        if blk0!=blk:
            maxlist.append([blk,layer,sub])
        if i== len(usedwqc)-1:
            maxlist.append([blk,layer,sub+1])
        if list[-2] not in input_num_list:
            input_num_list.append(list[-2])
    print(maxlist)
    verilog='''module MM_top_half
#(
parameter array_m = {}, 
parameter array_n = {},
parameter sic_n = {},
parameter data_width = {},
parameter log2_array_m = {},
parameter log2_array_n = {}
)
(
    input clk,
    input rst_n,
    input [16:0] state_in,
'''.format(w,h,sum(wlist),bits,math.ceil(math.log(w,2)),math.ceil(math.log(h,2)))
    for num in input_num_list:
        verilog +="\tinput [data_width*{}-1:0] MM_in_data{},\n".format(num,num)
    verilog +='''    output Out_data
);
//input [data_width*array_m-1:0] In_data,
//output [array_n*(data_width+3+log2_array_m)-1:0] Out_data
//wire [array_n*(data_width+3)-1:0]   MM_out_data;
wire [4:0] state_block;
wire [1:0] state_layer;
wire [9:0] state_sub;
//reg [data_width*array_m-1:0] MM_in_data;
//wire Out_data1, Out_data2;
'''
    verilog+='''
//reg [sic_n*data_width-1:0] SIC_in;
wire [sic_n*(2*data_width-1)-1:0] SIC_out;
//reg [array_m*array_n*(data_width+4)-1:0]    Matadd_in;
'''

    verilog +="parameter "
    for i in range(blocks):
        i+=1
        if i==(blocks):
            verilog+="b{}=5'b{:05b};\n".format(i,i)
        else:
            verilog+="b{}=5'b{:05b}, ".format(i,i)
    verilog +="parameter l0=2'b00,l1=2'b01,l2=2'b10;\n"
    verilog +="parameter "
    for i in range(subsmax):
            if i==(subsmax-1):
                verilog+="s{}=10'b{:010b};\n".format(i,i)
            else:
                verilog+="s{}=10'b{:010b}, ".format(i,i)
    for i in range(len(wlist)):
        i-=r
        if i!=0:
            if i < 0:
                sign = 'n'
            else:
                sign = 'p'   
            verilog+='''reg  [{}*data_width-1:0] SIC_in_{}{};\n'''.format(wlist[i+r],sign,abs(i))


    for list in usedwqc:
        for i in range(len(list)-2):
            if list[i]!=0:
                i-=r
                if i!=0:
                    if i < 0:
                        sign = 'n'
                    else:
                        sign = 'p'  
                    verilog+='''wire [{}*data_width-1:0] {}_out_{}{};\n'''.format(list[i+r],list[-1],sign,abs(i))



    verilog+='''
assign  state_block = state_in[16:12];
assign  state_layer = state_in[11:10]; 
assign  state_sub = state_in[9:0];
'''

    verilog+='''\nalways @(*) begin\n''' 

    Fblk = True
    Flayer = True
    # for list in usedwqc:
    for i in range(len(usedwqc)):
        list = usedwqc[i]
        name1 = list[-1].split('b')
        name2 = name1[1].split('c')
        blk = int(name2[0])
        name3 = name2[1].split('s')
        layer = int(name3[0])
        sub = int(name3[1])
        if i != len(usedwqc)-1:
            listn = usedwqc[i+1]
            name1n = listn[-1].split('b')
            name2n = name1n[1].split('c')
            blkn = int(name2n[0])
            name3n = name2n[1].split('s')
            layern = int(name3n[0])
            subn = int(name3n[1])
        if maxlist[blk-1][1]>0:
            if Fblk:
                if blk == 1:
                    verilog+='''\tcase(state_block)\n\tb{}: begin\n'''.format(blk)
                else:
                    verilog+='''\tb{}: begin\n'''.format(blk)
                Fblk = False
            if Flayer:
                if layer == 0:
                    verilog+='''\t\tcase(state_layer)\n\t\tl{}: begin\n'''.format(layer)
                else:
                    verilog+='''\t\tl{}: begin\n'''.format(layer)
                Flayer = False
            if sub == 0:
                verilog+='''\t\t\tcase(state_sub)\n\t\t\ts{}: begin\n'''.format(sub)
            else:
                verilog+='''\t\t\ts{}: begin\n'''.format(sub)
            for i in range(len(list)-2):
                if list[i]!=0:
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'                    
                        if wlist[i+r]-list[i+r]>0:
                            verilog+='''\t\t\tSIC_in_{}{} <= {{{}'b0,{}_out_{}{}}};\n'''.format(sign,abs(i),(wlist[i+r]-list[i+r])*bits,list[-1],sign,abs(i))
                        else:
                            verilog+='''\t\t\tSIC_in_{}{} <= {}_out_{}{};\n'''.format(sign,abs(i),list[-1],sign,abs(i))                     
                else:
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+='''\t\t\tSIC_in_{}{} <= 0;\n'''.format(sign,abs(i)) 
            verilog+='''\t\t\tend\n'''
            if subn == 0 or (layer == maxlist[blk-1][1] and sub == maxlist[blk-1][2] and blk==blocks):
                verilog+='''\t\t\tdefault:  begin\n'''
                for i in range(len(list)-2):
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+='''\t\t\tSIC_in_{}{} <= 0;\n'''.format(sign,abs(i)) 

                verilog+='''\t\t\tend\n\t\t\tendcase\n\t\tend\n'''
                Flayer = True
            if layer == maxlist[blk-1][1] and sub == maxlist[blk-1][2]:
                verilog+='''\t\tdefault:  begin\n'''
                for i in range(len(list)-2):
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+='''\t\tSIC_in_{}{} <= 0;\n'''.format(sign,abs(i)) 
                verilog+='''\t\tend\n\t\tendcase\n\tend\n'''
                Fblk = True
        else:
            if Fblk:
                if blk == 1:
                    verilog+='''\tcase(state_block)\nb{}: begin\n'''.format(blk)
                else:
                    verilog+='''\tb{}: begin\n'''.format(blk)
                Fblk = False
            if sub == 0:
                verilog+='''\t\tcase(state_sub)\n\t\ts{}: begin\n'''.format(sub)
            else:
                verilog+='''\t\ts{}: begin\n'''.format(sub)
            for i in range(len(list)-2):
                if list[i]!=0:
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'                    
                        if wlist[i+r]-list[i+r]>0:                     
                            verilog+='''\t\tSIC_in_{}{} <= {{{}'b0,{}_out_{}{}}};\n'''.format(sign,abs(i),(wlist[i+r]-list[i+r])*bits,list[-1],sign,abs(i))
                        else:
                            verilog+='''\t\tSIC_in_{}{} <= {}_out_{}{};\n'''.format(sign,abs(i),list[-1],sign,abs(i))                        
                else:
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+='''\t\tSIC_in_{}{} <= 0;\n'''.format(sign,abs(i)) 
            verilog+='''\t\tend\n'''
            if layer == maxlist[blk-1][1] and sub == maxlist[blk-1][2]:
                verilog+='''\t\tdefault:  begin\n'''
                for i in range(len(list)-2):
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+='''\t\tSIC_in_{}{} <= 0;\n'''.format(sign,abs(i)) 
                verilog+='''\t\tend\n\t\tendcase\n\tend\n'''
                Fblk = True           
        # if layer ==  maxlist[blk][1] and sub == maxlist[blk][2]:
    verilog+='''\tdefault:  begin\n'''
    for i in range(len(list)-2):
        i-=r
        if i < 0:
            verilog+='''\tSIC_in_n{} <= 0;\n'''.format(abs(i)) 
        elif i > 0:
            verilog+='''\tSIC_in_p{} <= 0;\n'''.format(i) 
    verilog+='''\tend\n\tendcase\nend\n'''
    

    verilog+="assign Out_data = ^SIC_out;\n"

# 例化SIC、Matadd、MM_wire
    for list in usedwqc:
        First = True
        verilog+='''{}_wire  {}_wire_u(.Data_in(MM_in_data{}),'''.format(list[-1],list[-1],list[-2])
        for i in range(len(list)-2):
            if First:
                if list[i]!=0:
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+='''.Data_out_{}{}({}_out_{}{})'''.format(sign,abs(i),list[-1],sign,abs(i))
                        First = False
            else:
                if list[i]!=0:
                    i-=r
                    if i!=0:
                        if i < 0:
                            sign = 'n'
                        else:
                            sign = 'p'  
                        verilog+=''', .Data_out_{}{}({}_out_{}{})'''.format(sign,abs(i),list[-1],sign,abs(i))
                        First = False 
        verilog+=''');\n'''             
    verilog+='''SIC    SIC_u('''
    for i in range(len(wlist)):
        i-=r
        if i!=0:
            if i < 0:
                sign = 'n'
            else:
                sign = 'p' 
            verilog+='''.Data_in_{}{}(SIC_in_{}{}),'''.format(sign,abs(i),sign,abs(i))       
    verilog+='''.Data_out(SIC_out));\n'''
    verilog+="endmodule"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)       
    with open(output_dir+"/MM_top_half.v","w") as f:
        f.write(verilog)



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

    wq = []
    shapelist = []
 
    for pname, p in qnn.named_parameters():
        if 'wq' in pname:
            logger.info('weight_param:{}, {}'.format(pname,p.size()))
            if p.ndimension() == 4:
                H,W,K1,K2 = p.shape
                if W*K1*K2!=27:
                    reshap_p = p.view(H,-1)
                    wq.append([pname,reshap_p.detach(),[H,W,K1,K2]]) 
                    shapelist.append([pname,[H,W,K1,K2]])
            elif p.ndimension() == 2:
                H,W = p.shape
                wq.append([pname,p.detach(),[H,W]])   
                shapelist.append([pname,[H,W]])

    blocks=3
    N = cfg.w_bit
    types = 2 ** N + 1 
    wise = 1
    Maxdis = [0]*types*wise


    wqc = []
    for [pname,p,p_shape] in wq:
        H,W = p.shape
        for i in range(int(W/wise)):
            subp = p[:,i*wise:(i+1)*wise]
            wqc.append([pname+'.{}'.format(i),subp])
    
    Output_dir = cfg.OUT_DIR
    wlist = cfg.wlist

    generate_sic(output_dir=Output_dir,bits=cfg.q_bit,wlist=wlist)
    USEDwqc=[]

    for p in wqc:
        usedw = generate_mmwire(p,bits=cfg.q_bit,output_dir=Output_dir,wlist=wlist)
        USEDwqc.append(usedw)

    
    generate_top_half(wqc[0],bits=cfg.q_bit,wlist=wlist,blocks=blocks,shapelist=shapelist,output_dir=Output_dir,usedwqc=USEDwqc)


if __name__ == '__main__':
    main()
