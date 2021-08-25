import argparse
import yaml
from easydict import EasyDict
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from pathlib import Path

def parse_config():
    parser = argparse.ArgumentParse(description='argparser')
    parser.add_argument('--paraname',default=None,type=str,help='help info',required=False,choice=[12,35,56])
    parser.add_argument('--knob',default=False, action='store_true',help='help info')


def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    #required表示这个参数必须给赋值
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')

    #choice表示输入参数的值只能是这几个
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')

    #设置action的作用相当于设置一个开关，在输入参数是加一个--sync_bn就可将他设置为True
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')

    #dest表示namespace中存储的名字是set_cfgs不是--set,相当于把--set关联到set_cfgs
    #nargs=argparse.REMAINDER. www.pynote.net/archives/1621
    #一般情况下，一个参数只能对应一个值，通过nargs实现一个参数可以有多个值, 输入的n个参数以list的形式存储 set_cfgs:[list]
    #argparse.REMAINDER表示生效的出入都作为这个参数的值，放到list里面
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    #cfg是pcdet定义的一个EasyDict的类.cfg = EasyDict()
    #https://blog.csdn.net/qq_41185868/article/details/103694833

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    #set_cfgss是parser最后一个参数--set/set_cfgs的值，意思就是整个后面的输入包括--，-类型都是--set的值
    if args.set_cfgs is not None:
        #把从vehicle232中取出来赋值给cfg的dict赋值给args.set_cfgs
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
        config.update(EasyDict(new_config))

    return config




if __name__ == '__main__':
    args, cfg = parge_config()
    # print('called with args:')
    # print('args:',args)
    # # print('cfg:',cfg)

    set_cfg = ['a',1,'b',2,'c',3]
    dict = zip(set_cfg[0::2],set_cfg[1::2])
    print(dict['a'])
    print(type(dict))

    # for k,v in zip(set_cfg[0::2],set_cfg[1::2]):
