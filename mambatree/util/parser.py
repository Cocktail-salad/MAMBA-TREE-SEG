import argparse
import yaml
import os.path as osp
from munch import Munch

def get_args(args):
# args对象很简单,它就是用来保存命令行输入内容的一个"容器"
    parser = argparse.ArgumentParser('tree_learn')
    # 使用了一个工具叫argparse,这个工具可以帮助程序正确解析命令行输入。首先程序用argparse创建一个对象,用来定义我们要读取哪些设置
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    if args is None:
        args = parser.parse_args()
    # 如果args为空(第一次进入函数),则调用parser的parse_args()方法
    # 这一方法会解析命令行实际输入的参数,然后把结果返回，结果是一个对象,赋值给args
    else:
        args = parser.parse_args(args)
    # 如果args不为空,表示不是第一次调用，则使用args对象本身作为基础,重新运行解析，将命令行新输入合并到当前args对象中
    return args


def load_yaml_file(filepath): # load_yaml_file函数的作用是读取YAML格式的配置文件,并解析文件内容返回
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def get_config(config_path):
    # Get the main configuration file 获取主配置文件
    main_cfg = load_yaml_file(config_path)

    # Get the default arguments, which are paths to other YAML files 获取默认参数，即其他 YAML 文件的路径
    default_args = main_cfg.pop('default_args', None)

    if default_args is not None:
        # Load the configuration from the default argument files 从默认参数文件加载配置
        for path in default_args:
            default_config = load_yaml_file(path)
            
            # Modify content of default args if specified so in main configuration and then update main configuration with modified default configuration
            # 如果在主配置中指定了默认参数，则修改默认参数的内容，然后用修改后的默认配置更新主配置
            for key in main_cfg:
                if key in default_config:
                    modify_default_cfg(default_config[key], main_cfg[key])
            
            main_cfg.update(default_config)
    return Munch.fromDict(main_cfg)


def get_args_and_cfg(args=None):
    args = get_args(args)
    cfg = get_config(args.config)
    print(args)
    if args.work_dir is not None:
        cfg.work_dir = osp.join('./work_dirs', args.work_dir)
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    return args, cfg


def modify_default_cfg(default_config, main_cfg):
    for key, value in main_cfg.items():
        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
            modify_default_cfg(default_config[key], value)
        else:
            default_config[key] = value
