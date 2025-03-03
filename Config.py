import argparse
import torch

def str2bool(v):
    """
    将字符串转换为布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    
    # 基础设置
    parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

    # 训练参数
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Max_train_steps', type=int, default=10e4, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

    # 模型参数
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
    parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size of training')
    parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before trianing')
    parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')


    args = parser.parse_args()
    args.dvc = torch.device(args.dvc)
    
    return args

# 环境名称配置
ENV_NAMES = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v4', 
             'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
BRIEF_ENV_NAMES = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']



hidden_sizes = [64,64]
cpu_processes = 8
num_threads = 2










