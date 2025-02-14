import os
import json
import torch
import pprint
import numpy as np
import random
from tensorboard_logger import Logger as TbLogger
import warnings
from options import get_options
import wandb

from problems.problem_tsp import TSP
from problems.problem_cvrp import CVRP
from agent.ppo import PPO

def load_problem(name):
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb and not opts.distributed:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    if not opts.no_wandb:
        wandb.init(project='NeuOpt', 
                    group=f'{opts.problem}{opts.graph_size}',
                    name= opts.agent + "_" + opts.run_name,
                    config=opts)
        
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)(
                            p_size = opts.graph_size,
                            init_val_met = opts.init_val_met,
                            with_assert = opts.use_assert,
                            DUMMY_RATE = opts.dummy_rate,
                            k = opts.k,
                            with_bonus = not opts.wo_bonus,
                            with_regular = not opts.wo_regular)
    
    # Figure out the RL algorithm
    if opts.agent == 'ppo':
        agent = PPO(problem, opts)
        expert = None
    elif opts.agent == 'gfn':
        from agent.gfn import GFN
        agent = GFN(problem, opts)
        expert = PPO(problem, opts) if opts.guided else None
    elif opts.agent == 'mle':
        from agent.mle import MLE
        agent = MLE(problem, opts)
        expert = PPO(problem, opts)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)
    
    if expert is not None:
        expert.load(f'pre-trained/{opts.problem}{opts.graph_size}.pt', actor_only=True)

    # Do validation only
    if opts.eval_only:
        # Load the validation datasets
        agent.start_inference(problem, opts.val_dataset, tb_logger)
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1
    
        # Start the actual training loop
        agent.start_training(problem, opts.val_dataset, tb_logger, expert=expert)
    
    if (not opts.no_wandb):
        wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    warnings.filterwarnings("ignore")
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(get_options())
