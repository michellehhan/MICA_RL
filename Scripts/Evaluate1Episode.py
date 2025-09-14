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

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from CodeColab.models.PPO import PPO
import CodeColab.UE.ElbowFixTarget.Scripts.utils as utils

device = torch.device('cpu')
work_dir = 'CodeColab/UE/ElbowFixTarget/Exp2/'

def EvaluateWithoutExo(weight,target,epi_len,sarc=False,registered=False):
    exo  = False
    suffix  = ""
    run_id = 'tgt_21' # use this to identify the run parameters
    p = utils.init_parameters()
    # Load Healthy Policy
    hlthy_state_dim = 9
    hlthy_action_dim = 6
    hlthy_env_name = utils.get_env_prefix(False,False)+p['new_model_nm']
    hlthy_policy_path = utils.init_policypath(hlthy_env_name,run_id,p)
    hlthy_agent = PPO(hlthy_state_dim, hlthy_action_dim, 0, 0, 1, 1, 0.2, True, 0.02)
    hlthy_agent.load(hlthy_policy_path)
    # 
    env_name_2 = ""
    if not(registered):
        env_name_2 = utils.register_env(p,exo,sarc)
    else:
        env_name_2= utils.get_env_prefix(exo,sarc)+p['new_model_nm']

    env = gym.make(env_name_2)
    state = env.reset()
    env.env.sim.model.body_mass[5] = weight *1.0
    env.env.sim_obsd.model.body_mass[5] = weight * 1.0
    env.env.sim.data.qpos[0] =0
    env.env.sim.forward()
    state[0]= env.env.sim.data.qpos[0]
    state[1]= env.env.sim.data.qvel[0]
    obs= np.concatenate((state[:2],state[0]-target,state[3:]),axis=None)
    data_store =[]
    frames = []
    for t in range(1, epi_len+1):
        frame = env.sim.renderer.render_offscreen(camera_id=0)
        frames.append(frame)
        obs[0]= obs[0]*2.1
        obs[3:] = np.clip(obs[3:]*2.1,0,1)
        print(obs)
        mus_action =  hlthy_agent.select_action(obs)
        mus_action =utils.scaleAction(mus_action,0,1)
        action = np.append(0, mus_action) # exo action is zero for hlthy
        
        state, _, _, _ = env.step(action)
        error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
        obs = np.concatenate((state[:2],error,state[3:]),axis=None)#np.append(state,target_angle) # this is 9 dimension does not include exo
        data_store.append({'plt_dict': utils.get_plot_data(env,exo,error).copy()})

    env.close()
    return data_store, frames


def EvaluateWithExo(ctrl_value,weight,target,epi_len,registered=False):
    exo, sarc = True, True
    suffix  = "ctrl_"+str(ctrl_value)
    run_id = 'tgt_21' # use this to identify the run parameters
    p = utils.init_parameters()
    # Load Healthy Policy
    hlthy_state_dim = 9
    hlthy_action_dim = 6
    hlthy_env_name = utils.get_env_prefix(False,False)+p['new_model_nm']
    hlthy_policy_path = utils.init_policypath(hlthy_env_name,run_id,p)
    hlthy_agent = PPO(hlthy_state_dim, hlthy_action_dim, 0, 0, 1, 1, 0.2, True, 0.02)
    hlthy_agent.load(hlthy_policy_path)
    # Load Exo Policy
    env_name_2 = ""
    if not(registered):
        env_name_2 = utils.register_env(p,exo,sarc)
    else:
        env_name_2= utils.get_env_prefix(exo,sarc)+p['new_model_nm']

    exo_state_dim = 10
    exo_action_dim = 1
    exo_env_name = utils.get_env_prefix(exo,sarc)+p['new_model_nm']
    exo_policy_path = utils.init_policypath(env_name_2,run_id,p,suffix="ctrl_"+str(10.0))
    exo_agent = PPO(exo_state_dim, exo_action_dim, 0, 0, 1, 1, 0.2, True, 0.02)
    exo_agent.load(exo_policy_path)

    env = gym.make(env_name_2)
    state = env.reset()

    env.env.sim.model.actuator('Exo').ctrllimited = True
    env.sim.model.actuator('Exo').ctrlrange = np.array((-1*ctrl_value,ctrl_value))
    env.env.sim.model.body_mass[5] = weight *1.0
    env.env.sim_obsd.model.body_mass[5] = weight * 1.0
    env.env.sim.data.qpos[0] =0
    env.env.sim.forward()
    state[0]= env.env.sim.data.qpos[0]
    state[1]= env.env.sim.data.qvel[0]
    exo_l = 0.0#env.sim.data.actuator('Exo').length.item(0)
    # obs[2] contains the error
    obs= np.concatenate((state[:2],state[0]-target,state[3:],exo_l),axis=None)
    data_store = []
    frames = []
    for t in range(1, epi_len+1):
        frame = env.sim.renderer.render_offscreen(camera_id=0)
        frames.append(frame)
        mus_action =  hlthy_agent.select_action(obs[:9])
        mus_action = utils.scaleAction(mus_action,0,1)
        exo_action  = exo_agent.select_action(obs) # exo_action is bidirectional [-1,1]
        exo_action = utils.scaleAction(exo_action,-ctrl_value,ctrl_value) # scale to desired control range
        action = np.append(exo_action, mus_action) # exo action is zero for hlthy
        #action = np.append(hlthy_action,exo_action[0])
        state, _, _, _ = env.step(action)
        error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
        exo_l = 0.0#env.sim.data.actuator('Exo').length.item(0)
        obs = np.concatenate((state[:2],error,state[3:],exo_l),axis=None)#np.append(state,target_angle) # this is 9 dimension does not include exo
        data_store.append({'plt_dict': utils.get_plot_data(env,exo,error).copy()})

    env.close()
    return data_store, frames

def frames2Video(filepath, frames):
    #os.makedirs(dirpath, exist_ok=True)
    # make a local copy
    skvideo.io.vwrite(filepath, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
    #show_video(filepath)

def calculate_energy(env,exo):
    a = abs(env.sim.data.actuator_force)
    exo_energy = np.zeros((1))
    if exo:
        exo_energy[0]= env.sim.data.actuator('Exo').force
    return np.linalg.norm(a), np.linalg.norm(exo_energy)


def plotEpisodeDataCombined(resultsdataset,ctrl_value,weight):

    exo_unpacked_data = utils.unpack_plot_dict(resultsdataset['exo'])
    hlthy_unpacked_data = utils.unpack_plot_dict(resultsdataset['healthy'])
    sarc_unpacked_data = utils.unpack_plot_dict(resultsdataset['sarc'])
    l_exo = 'Exo_'+str(ctrl_value)+"_wt_"+str(weight)
    l_hlthy='Healthy'+"_wt_"+str(weight)
    l_sarc = 'Sarc'+"_wt_"+str(weight)

    x = np.arange(len(exo_unpacked_data['r_elbow_torque']))*0.01

    plt.figure(1)
    plt.plot(x,exo_unpacked_data['r_elbow_torque'],label=l_exo)
    plt.plot(x,hlthy_unpacked_data['r_elbow_torque'],label=l_hlthy)
    plt.plot(x,sarc_unpacked_data['r_elbow_torque'],label=l_sarc)
    plt.xlabel("Episode Time t (seconds)")
    plt.ylabel("Torque (Nm)")
    plt.ylim([-ctrl_value*1.2,ctrl_value*1.2])
    plt.title("Right Elbow Joint Torque")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(work_dir+"plots/JointTorques.png")
    plt.close()

    plt.figure(2)
    plt.plot(x,exo_unpacked_data['angle_error']+0.1,label=l_exo)
    plt.plot(x,hlthy_unpacked_data['angle_error']+0.1,label=l_hlthy)
    plt.plot(x,sarc_unpacked_data['angle_error']+0.1,label=l_sarc)
    plt.xlabel("Episode Time t (seconds)")
    plt.ylabel("Joint Angle Error (rad)")
    plt.title("Right Elbow Joint Angle Error")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(work_dir+"plots/ElbowFlexAngle.png")
    plt.close()

    fig, axs = plt.subplots(3, 2,sharex=True, sharey=True)
    fig.suptitle('Right Arm Muscle Excitations')
    plt.subplots_adjust(left=0.1,bottom=0.2,right=0.95,top=0.9, wspace=0.1,hspace=0.25)
    index  = 0 
    labels = [l_exo,l_hlthy,l_sarc]
    for i in range(1,len(exo_unpacked_data['act_ctrl'][0,:])):
        l = str(exo_unpacked_data['act_headers'][i])
        r = index//2
        c = index % 2
        pe=axs[r,c].plot(x,exo_unpacked_data['act_ctrl'][:,i])
        ph=axs[r,c].plot(x,hlthy_unpacked_data['act_ctrl'][:,i])
        ps=axs[r,c].plot(x,sarc_unpacked_data['act_ctrl'][:,i])
        axs[r,c].set_title('m ='+l,fontsize='small')
        axs[r,c].set_xlim([0, 0.5])
        if index >=4:
            axs[r,c].set_xlabel('Episode Time t (seconds)')
        index +=1 
    fig.supylabel('Muscle Excitation $e_m(t)$')
    fig.legend([pe, ph, ps], labels=labels, loc="outside lower right", mode = "expand", ncol=3) 
    plt.savefig(work_dir+"plots/ExcitationsControlsRightArm.png")
    plt.close()

    fig, axs = plt.subplots(3, 2,sharex=True, sharey=True)
    fig.suptitle('Right Arm Muscle Activations')
    plt.subplots_adjust(left=0.1,bottom=0.2,right=0.95,top=0.9,  wspace=0.1,hspace=0.25)
    index  = 0 
    labels = [l_exo,l_hlthy,l_sarc]
    for i in range(1,len(exo_unpacked_data['activation'][0,:])):
        l = str(exo_unpacked_data['act_headers'][i])
        r = index//2
        c = index % 2
        pe=axs[r,c].plot(x,exo_unpacked_data['activation'][:,i])
        ph=axs[r,c].plot(x,hlthy_unpacked_data['activation'][:,i])
        ps=axs[r,c].plot(x,sarc_unpacked_data['activation'][:,i])
        axs[r,c].set_title('m ='+l,fontsize='small')
        axs[r,c].set_xlim([0, 0.5])
        if index >=4:
            axs[r,c].set_xlabel('Episode Time t (seconds)')
        index +=1 
    fig.supylabel('Muscle Activation $a_m(t)$')
    fig.legend([pe, ph, ps], labels=labels, loc="outside lower right", mode = "expand", ncol=3) 
    plt.savefig(work_dir+"plots/MuscleActivationsRightArm.png")
    plt.close()


    plt.figure(3)
    plt.plot(x,exo_unpacked_data['act_ctrl'][:,0],label=l_exo)
    plt.xlabel("Episode Time t (seconds)")
    plt.ylabel("Actuation")
    plt.legend(loc='lower right')
    plt.title("Exo Actuations During an Episode")
    plt.grid(True)
    plt.savefig(work_dir+"plots/ExoActuation.png")
    plt.close()

def plotEpisodeData(data_store,type,weight,env_name_2 = 'eblow-v0'):
    unpacked_data = utils.unpack_plot_dict(data_store)
    plt.figure(1)
    plt.plot(unpacked_data['r_elbow_torque'],label='r_elbow_torque')
    plt.xlabel("Episode Time")
    plt.ylabel("Right Elbow Joint Torque")
    plt.title("Joint Torque ("+type+")")
    plt.legend(loc='lower right')
    plt.savefig(work_dir+"plots/JointTorques_"+type+".png")
    plt.close()

    plt.figure(2)
    plt.plot(unpacked_data['angle_error'],label='angle_error')
    plt.xlabel("Episode Time")
    plt.ylabel("Right Elbow Joint Angle Error")
    plt.title("Elbow Angle Error ("+type+")")
    plt.legend(loc='lower right')
    plt.savefig(work_dir+"plots/ElbowFlexAngle_"+type+".png")
    plt.close()

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Muscle Excitations for Right Arm ('+type+")")
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9, wspace=0.6,hspace=0.4)
    index  = 0 

    for i in range(1,len(unpacked_data['act_ctrl'][0,:])):
        l = str(unpacked_data['act_headers'][i])
        r = index//2
        c = index % 2
        axs[r,c].plot(unpacked_data['act_ctrl'][:,i],label=l)
        axs[r,c].set_title(l)
        index +=1
    plt.savefig(work_dir+"plots/MuscleExcitationsRightArm_"+type+".png")
    plt.close()

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Muscle Activations for Right Arm ('+type+")")
    plt.subplots_adjust(left=0.1,bottom=0.12,right=0.95,top=0.9, wspace=0.3,hspace=0.5)
    index  = 0 
    for i in range(1,len(unpacked_data['activation'][0,:])):
        l = str(unpacked_data['act_headers'][i])
        r = index//2
        c = index % 2
        pe=axs[r,c].plot(unpacked_data['activation'][:,i])
        axs[r,c].set_title(l)
        #axs[r,c].set_ylim([-1, 1])
        index +=1 
    plt.savefig(work_dir+"plots/MuscleActivationsRightArm.png")
    plt.close()

    if type.startswith("Exo"):
        plt.figure(3)
        plt.plot(unpacked_data['act_ctrl'][:,0],label=unpacked_data['act_headers'][i])
        plt.xlabel("Episode Time")
        plt.ylabel("Exo Actuation")
        plt.legend(loc='lower right')
        plt.title("Exo Actuations ("+type+")")
        plt.savefig(work_dir+"plots/ExoActuation"+type+".png")
        plt.close()
    

resultsdataset= {}
target =1.0
weight = 5.0
ctrl_value = 11.5
epi_len = 500
data_store, frames = EvaluateWithExo(ctrl_value,weight,target,epi_len,registered=False)
resultsdataset['exo']=data_store
frames2Video(work_dir+'videos/Exo_ctrl_'+str(ctrl_value)+'.mp4', frames)

data_store, frames = EvaluateWithoutExo(weight,target,epi_len,sarc=False,registered=False)
resultsdataset['healthy']=data_store
frames2Video(work_dir+'videos/Healthy.mp4', frames)

data_store, frames = EvaluateWithoutExo(weight,target,epi_len,sarc=True,registered=False)
resultsdataset['sarc']=data_store
frames2Video(work_dir+'videos/sarc.mp4', frames)

plotEpisodeDataCombined(resultsdataset,ctrl_value,weight)
##ctrls = [1000.0,900.0,800.0,700.0,600.0,500.0,400.0,300.0,200.0,100.0]
#for ctrl in ctrls:
#    EvaluateHeightWithExo(ctrl,run_id = str(4))