import numpy as np
import skvideo.io
import os, pickle, glob, time, math
import gym
import matplotlib.pyplot as plt
import myosuite
import mujoco
from datetime import datetime
from base64 import b64encode
from IPython.display import HTML
import sys
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from myosuite.envs.env_variants import register_env_variant

from CodeColab.models.PPO import PPO
import CodeColab.UE.ElbowFixTarget.Scripts.utils as utils

device = torch.device('cpu')

env_name = 'myoElbowPose1D6MExoRandom-v0'

def main(env_name,exo, sarc, train_steps, run_id, weight_value, target_value):
    exo = False
    sarc = False
    #########################  Change the following to match your run
    p = utils.init_parameters(train_steps=train_steps)
    retrainExistingPolicy = False

    env_name_2 = utils.register_env(p,exo,sarc)
    env = gym.make(env_name_2)
    env.reset()
    ################################### Training ###################################

    print("training environment name : " + env_name_2)

    # dimensions
    state_dim = 9 # Joint angle and velocity
    action_dim = 6 # muscle actions without Exo

    # initialize a PPO agent default action is within [0,1]
    ppo_agent = PPO(state_dim, action_dim, p['lr_actor'], p['lr_critic'], p['gamma'], p['K_epochs'], p['eps_clip'], p['has_continuous_action_space'], p['action_std'])
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    log_f_name = utils.init_logpath(env_name_2,run_id,p)
    checkpoint_path = utils.init_policypath(env_name_2,run_id,p)
    if retrainExistingPolicy:
        ppo_agent.load(checkpoint_path)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    # printing and logging variables
    print_running_reward,print_running_episodes = 0,0
    log_running_reward,log_running_episodes = 0,0
    time_step,i_episode = 0,0

    # training loop
    while time_step <= p['max_training_timesteps']: 
        state = env.reset()
        if weight_value=="-1":
            weight = np.random.choice(np.arange(1,6),1) # between 1 to 5 kg
        else:
            weight = float(weight_value)
        
        if target_value=="-1":
            target = np.random.choice(np.arange(5,22),1)*0.1
        else:
            target = float(target_value)

        #print(target, weight)
        env.env.sim.model.body_mass[5] = weight *1.0
        env.env.sim_obsd.model.body_mass[5] = weight * 1.0
        env.env.sim.data.qpos[0]=0 # angle error
        env.env.sim.forward()
        state[0]= env.env.sim.data.qpos[0]
        state[1]= env.env.sim.data.qvel[0]
        #state[2]= state[0].copy()-target
        current_ep_reward = 0
        # obs[2] contains the error
        obs= np.concatenate((state[:2],state[0]-target,state[3:]),axis=None)
        done = False
        for t in range(1, p['max_ep_len']+1):
            mus_action =  ppo_agent.select_action(obs)
            mus_action =utils.scaleAction(mus_action,0,1)
            action = np.append(0, mus_action) # exo action is zero for hlthy
        
            state, _, _, _ = env.step(action)
            error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
            reward = utils.get_reward(obs[2],error) # obs[2] is the old error
            obs = np.concatenate((state[:2],error,state[3:]),axis=None)#np.append(state,target_angle) # this is 9 dimension does not include exo
            ppo_agent.buffer.rewards.append(reward)
            if t==p['max_ep_len']:
                done = True
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % p['update_timestep'] == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if p['has_continuous_action_space'] and time_step % p['action_std_decay_freq'] == 0:
                ppo_agent.decay_action_std(p['action_std_decay_rate'], p['min_action_std'])

            # log in logging file
            if time_step % p['log_freq'] == 0:
                log_running_reward, log_running_episodes  = utils.printLogs(log_f, log_running_reward, log_running_episodes,i_episode, time_step)

            # printing average reward
            if time_step % p['print_freq'] == 0:
                print_running_reward, print_running_episodes = utils.printReward(print_running_reward,print_running_episodes,i_episode, time_step)
                
            # save model weights
            if time_step % p['save_model_freq'] == 0:
                ppo_agent.save(checkpoint_path)
                utils.printModelSaved(checkpoint_path,start_time)
            # break; if the episode is over
            #if done:
            #    break
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    ppo_agent.save(checkpoint_path)
    log_f.close()
    env.close()

    # print total training time
    print("path", checkpoint_path)
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


#########################  INPUT Arguments (RUN_ID, WEIGHT, TARGET, TRAIN_STEPS )
run_id = sys.argv[1] # use this to identify the run parameters
weight_value  = sys.argv[2] 
target_value  = sys.argv[3] 
train_steps  = sys.argv[4] 

main(env_name, exo, sarc, train_steps, run_id, weight_value, target_value)