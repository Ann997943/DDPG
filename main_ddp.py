import os
import shutil
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import gymnasium as gym
import Config
from datetime import datetime
from utils import evaluate_policy
from DDPG import DDPG_agent
from Config import get_args, ENV_NAMES, BRIEF_ENV_NAMES
from torch.nn.parallel import DistributedDataParallel as DDP

# 销毁分布式进程组
def cleanup():
    dist.destroy_process_group()

# 获取参数
opt = get_args()
# print(opt)

torch.set_num_threads(Config.num_threads)

iteration_time_sum = 0
iteration_count = 0 
dc_time_sum = 0 
mu_time_sum = 0


def main(rank, world_size):
    

    #set up ddp environment
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    global iteration_time_sum, dc_time_sum, mu_time_sum, iteration_count
    
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']

    # Build Env
    env_seed = opt.seed + rank*1000
    env = gym.make(EnvName[opt.EnvIdex], render_mode = None)  # 训练时关闭渲染
    eval_env = gym.make(EnvName[opt.EnvIdex])
    # env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    
    
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】

    if rank == 0:
        print(opt)

    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
        f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')
        

    # Seed Everything
    torch.manual_seed(env_seed)
    torch.cuda.manual_seed(env_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Build SummaryWriter to record training curves
    if opt.write and rank==0:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): 
        os.mkdir('model')

    agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
    agent.actor = DDP(agent.actor, broadcast_buffers=False)
    agent.q_critic = DDP(agent.q_critic, broadcast_buffers=False)
    
    

    if opt.Loadmodel: 
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)


    # if opt.render:
    #         while True:
    #             score = evaluate_policy(env, agent, turns=1)
    #             if rank==0:
    #                 print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    
    total_steps = 0
    iteration_start_time = time.perf_counter()  # 记录每次迭代开始时间
    dc_start_time = time.perf_counter()  # 记录DC部分开始时间

    while total_steps < opt.Max_train_steps:

        # print(f"Process rank: {rank} Seed:{env_seed}")
        s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
        env_seed += 1
        done = False

        '''Interact & train'''
        
        while not done:  

            if total_steps < opt.random_steps: 
                a = env.action_space.sample()
            else: 
                a = agent.select_action(s, deterministic=False)
            s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
            done = (dw or tr)

            agent.replay_buffer.add(s, a, r, s_next, dw)
            s = s_next
            total_steps += 1

            '''train'''
            
            if total_steps % 512 ==0:
                dist.barrier()
                
                if rank == 0:
                    #starting time
                    dc_end_time = time.perf_counter()  
                    dc_time_sum += (dc_end_time - dc_start_time) 
                    mu_start_time = time.perf_counter()  

                dist.barrier()
                agent.train()
                dist.barrier()
            
                if rank == 0:
                    iteration_count += 1
                    mu_end_time = time.perf_counter()  
                    mu_time_sum += (mu_end_time - mu_start_time) 
                    iteration_end_time = time.perf_counter() 
                    iteration_time_sum += (iteration_end_time - iteration_start_time) 
            
                
                
                dist.barrier()
                iteration_start_time = time.perf_counter()
                dc_start_time = time.perf_counter()

            
    

            '''record & log'''
            if total_steps % opt.eval_interval == 0 and rank==0:
                ep_r = evaluate_policy(eval_env, agent, turns=3)
                if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k')

            '''save model'''
            if rank==0 and total_steps % opt.save_interval == 0:
                agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))

            
    if rank==0:       
        print(f"Total DC Time (Environment Interaction): {dc_time_sum} s")
        print(f"Total MU Time (Model Update): {mu_time_sum} s")
        print(f"Total Iteration Time: {iteration_time_sum} s")
        print(dc_time_sum+mu_time_sum)
        print("Average Execution Time per Iteration: {} s".format(iteration_time_sum/(iteration_count)))
        print("Iteration: {} ".format(iteration_count))
        print("Average Execution Time per Step: {} s".format(iteration_time_sum/10e4))
    env.close()
    eval_env.close()
     #cleanup()
    

    

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    world_size = Config.cpu_processes
    mp.spawn(main, 
            args=(world_size,), 
            nprocs=world_size, 
            join=True
    )
   


