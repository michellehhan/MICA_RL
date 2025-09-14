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
import CodeColab.UE.ElbowFixTarget.Scripts.utils as utils
from CodeColab.models.PPO import PPO
device = torch.device('cpu')

## This function uses EXO policy trained for CTRL_VALUE =10
## and modulate it for other ctrl_values
##

def calculate_energy(env,exo):
    a = abs(env.sim.data.actuator_force)
    exo_energy = np.zeros((1))
    if exo:
        exo_energy= env.sim.data.actuator('Exo').force
    return np.linalg.norm(a), np.linalg.norm(exo_energy)

model_dir = 'CodeColab/MyoAssist/Model/'
work_dir = 'CodeColab/UE/ElbowFixTarget/'
env_name='myoElbowPose1D6MExoRandom-v0'

def EvaluateAngleWithoutExo(sarc=False,registered=False):
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
    
    env_name_2 = ""
    if not(registered):
        env_name_2 = utils.register_env(p,exo,sarc)
    else:
        env_name_2= utils.get_env_prefix(exo,sarc)+p['new_model_nm']
   
    data_f = open(work_dir+'ResultsData/'+env_name_2+'_'+run_id+suffix+'.csv',"w")

    env = gym.make(env_name_2)
    env.reset()
    num_eps = 200
    target = 2.1
    for weight in range(1,6):
        print(str(weight))
        for num_ep in range(num_eps):
            state = env.reset()
            env.env.sim.model.body_mass[5] = weight *1.0
            env.env.sim_obsd.model.body_mass[5] = weight * 1.0
            env.env.sim.data.qpos[0] =0
            env.env.sim.forward()
            state[0]= env.env.sim.data.qpos[0]
            state[1]= env.env.sim.data.qvel[0]
            obs= np.concatenate((state[:2],state[0]-target,state[3:]),axis=None)
            done = False
            min_error = target
            time_to_min = 0
            energy = 0
            exoenergy = 0
            max_energy =0
            for t in range(1, p['max_ep_len']+1):
                mus_action =  hlthy_agent.select_action(obs)
                mus_action = utils.scaleAction(mus_action, 0, 1)
                action = np.append(0, mus_action) # exo action is zero for hlthy
                state, _, _, _ = env.step(action)
                error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
                obs = np.concatenate((state[:2],error,state[3:]),axis=None)#np.append(state,target_angle) # this is 9 dimension does not include exo
                m_e, e_e = calculate_energy(env,False)
                energy+=m_e
                if abs(error) < min_error:
                    min_error = abs(error)
                    time_to_min = t
                    max_exoenergy = exoenergy
                    max_energy = energy

            data_f.write(str(target)+","+str(weight)+","+str(min_error)+','+
                        str(time_to_min)+','+
                        str(energy)+','+
                        str(exoenergy)+'\n')
    data_f.close()
    env.close()

def EvaluateAngleWithExo(ctrl_value,registered=False):
    exo, sarc  = True, True
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
    exo_policy_path = utils.init_policypath(env_name_2,run_id,p,suffix="ctrl_"+str(20.0))
    exo_agent = PPO(exo_state_dim, exo_action_dim, 0, 0, 1, 1, 0.2, True, 0.02)
    exo_agent.load(exo_policy_path)
   
    data_f = open(work_dir+'ResultsData/Exo10_'+env_name_2+'_'+run_id+suffix+'.csv',"w")

    env = gym.make(env_name_2)
    env.reset()
    env.env.sim.model.actuator('Exo').ctrllimited = True
    env.sim.model.actuator('Exo').ctrlrange = np.array((-1*ctrl_value,ctrl_value))

    num_eps = 200
    target = 2.1
    for weight in range(1,6):
        print(str(target),str(weight),str(ctrl_value))
        for num_ep in range(num_eps):
            state = env.reset()
            env.env.sim.model.body_mass[5] = weight *1.0
            env.env.sim_obsd.model.body_mass[5] = weight * 1.0
            env.env.sim.data.qpos[0] =0
            env.env.sim.forward()
            state[0]= env.env.sim.data.qpos[0] 
            state[1]= env.env.sim.data.qvel[0]
            exo_l = 0.0#env.sim.data.actuator('Exo').length.item(0)
            # obs[2] contains the error
            obs= np.concatenate((state[:2],state[0]-target,state[3:],exo_l),axis=None)
            done = False
            min_error = target
            time_to_min = 0
            energy = 0
            exoenergy = 0
            max_energy =0
            for t in range(1, p['max_ep_len']+1):
                mus_action =  hlthy_agent.select_action(obs[:9])
                mus_action = utils.scaleAction(mus_action,0,1)
                exo_action  = exo_agent.select_action(obs) # exo_action is bidirectional [-1,1]
                exo_action = utils.scaleAction(exo_action,-ctrl_value,ctrl_value) # scale to desired control range
                action = np.append(exo_action, mus_action) # exo action is zero for hlthy
                state, _, _, _ = env.step(action)
                error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
                exo_l = 0.0#env.sim.data.actuator('Exo').length.item(0)
                obs = np.concatenate((state[:2],error,state[3:],exo_l),axis=None)#np.append(state,target_angle) # this is 9 dimension does not include exo
                m_e, e_e = calculate_energy(env,True)
                energy+=m_e
                exoenergy+=e_e
                if abs(error) < min_error:
                    min_error = abs(error)
                    time_to_min = t
                    max_exoenergy = exoenergy
                    max_energy = energy

            data_f.write(str(target)+","+str(weight)+","+str(min_error)+','+
                        str(time_to_min)+','+
                        str(energy)+','+
                        str(exoenergy)+'\n')

    data_f.close()
    env.close()
    return True


def plotWeightVsMinAngleErrorCTRLDes(ctrls,env_name = 'elbow-v0' ,run_id = 'tgt_21'):
    ctrl_type = ['ctrl_'+str(s) for s in ctrls]
    ctrl_type.extend(['sarc','hlthy'])
    weights = np.array([1,2,3,4,5])
    error = np.zeros((len(ctrl_type),len(weights))) # absolute error in reaching the target
    delay = np.zeros((len(ctrl_type),len(weights)))
    energy = np.zeros((len(ctrl_type),len(weights)))
    exoenergy = np.zeros((len(ctrl_type),len(weights)))
    for j in range(len(ctrl_type)):
        prefix = 'exosarc'
        suffix = ctrl_type[j]
        if ctrl_type[j]=='hlthy':
            prefix = ''
            suffix = ''
        if ctrl_type[j]=='sarc':
            prefix = 'sarc'
            suffix = ''
        env_name2 = prefix+env_name+'_'+run_id+suffix
        fileprefix = ""
        if ctrl_type[j].startswith("ctrl"):
            fileprefix="Exo10_"
        a = np.loadtxt(work_dir+"ResultsData/"+fileprefix+env_name2+'.csv', delimiter=',')
        for i  in range(len(weights)):
            error[j,i]=np.average(a[np.where(a[:,1]==weights[i]),2])
            delay[j,i]=np.average(a[np.where(a[:,1]==weights[i]),3])
            energy[j,i]=np.average(a[np.where(a[:,1]==weights[i]),4])
            exoenergy[j,i]=np.average(a[np.where(a[:,1]==weights[i]),5])
        plt.figure(1)
        plt.plot(weights,error[j,:],label=ctrl_type[j])
        plt.figure(2)
        plt.plot(weights,delay[j,:],label=ctrl_type[j])
    
    
    plt.figure(1)
    plt.xlabel("Carried Weight")
    plt.ylabel("Flex Angle Error From Target (Rad)")
    plt.legend(loc='lower left')
    plt.savefig(work_dir+"plots/CTRLDesMinErrorResults_Weights_"+run_id+".png")
    plt.close()
    plt.figure(2)
    plt.xlabel("Carried Weight")
    plt.ylabel("Average Time Steps to Reach Minimum Flex Angle Error")
    plt.legend(loc='lower right')
    plt.savefig(work_dir+"plots/CTRLDesMaxDelayResults_Weights_"+run_id+".png")
    plt.close()
    plt.figure(3)
    exo_avg = np.average(exoenergy[:,:],axis = 1);
    mus_avg = np.average(energy[:,:],axis = 1);
    plt.scatter(ctrl_type,exo_avg,label='Exo Actuation')
    plt.scatter(ctrl_type,mus_avg,label='Muscle Activation')
    plt.legend(loc='upper left')
    plt.xlabel("Activation - Energy")
    plt.ylabel("Control Scenario")
    plt.savefig(work_dir+"plots/CTRLDesEnergyCompareResults_Weights_"+run_id+".png")
    plt.close()

def plotErrorMinCTRLDes(ctrls,env_name = 'elbow-v0' ,run_id = 'tgt_21'):
    ctrl_range = ctrls
    ctrl_type = ['ctrl_'+str(s) for s in ctrl_range]
    ctrl_type.extend(['sarc','hlthy'])
    error = np.zeros((len(ctrl_type),1))
    delay = np.zeros((len(ctrl_type),1))
    energy = np.zeros((len(ctrl_type),1))
    exoenergy = np.zeros((len(ctrl_type),1))
    for j in range(len(ctrl_type)):
        prefix = 'exosarc'
        suffix = ctrl_type[j]
        if ctrl_type[j]=='hlthy':
            prefix = ''
            suffix = ''
        if ctrl_type[j]=='sarc':
            prefix = 'sarc'
            suffix = ''
        env_name2 = prefix+env_name+'_'+run_id+suffix
        fileprefix = ""
        if ctrl_type[j].startswith("ctrl"):
            fileprefix="Exo10_"
        a = np.loadtxt(work_dir+'ResultsData/'+fileprefix+env_name2+'.csv', delimiter=',')      
        error[j]=np.average(a[:,2])
        delay[j]=np.average(a[:,3])
        energy[j]=np.average(a[:,4])
        exoenergy[j]=np.average(a[:,5])
    
    plt.figure(1)
    plt.scatter(ctrl_type,error)
    plt.xlabel("Control Scenario")
    plt.ylabel("Minimum Error Angle (rad)")
    plt.savefig(work_dir+"plots/CTRLDesMinErrorResults_"+run_id+".png")

    plt.figure(2)
    plt.scatter(ctrl_type,delay)
    plt.xlabel("Control Scenario")
    plt.ylabel("Time to Reach Minimum Angle Error from Target ")
    plt.savefig(work_dir+"plots/CTRLDesDelayResults_"+run_id+".png")

    plt.figure(3)
    plt.scatter(ctrl_type,energy)
    plt.scatter(ctrl_type,exoenergy)
    plt.xlabel("Control Scenario")
    plt.ylabel("Energy Comsumption ")
    plt.savefig(work_dir+"plots/CTRLDesEnergyResults_"+run_id+".png")
    plt.close()

#EvaluateAngleWithoutExo(sarc=False)
#EvaluateAngleWithoutExo(sarc=True)
ctrls = [5.0,10.0,15.0,20.0,25.0]
#registered = False
#for ctrl in ctrls:
#    registered = EvaluateAngleWithExo(ctrl,registered)
plotWeightVsMinAngleErrorCTRLDes(ctrls)
plotErrorMinCTRLDes(ctrls)

