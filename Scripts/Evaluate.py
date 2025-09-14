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
from scipy.stats import t as ttest

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import CodeColab.UE.ElbowFixTarget.Scripts.utils as utils
from CodeColab.models.PPO import PPO
device = torch.device('cpu')

## This function uses EXO policy trained for next higher ctrl range
## and modulate it to obtain desired control range
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

def EvaluateAngleWithoutExo(epi_len, sarc=False,registered=False):
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
            for t in range(1, epi_len+1):
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
                        str(time_to_min)+','+ str(energy)+','+
                        str(exoenergy)+','+ str(error)+'\n')
    data_f.close()
    env.close()

def EvaluateAngleWithExo(ctrl_value, trained_ctrls,epi_len,registered=False):
    exo, sarc  = True, True
    suffix  = "ctrl_"+str(ctrl_value)
    run_id = 'tgt_21' # use this to identify the run parameters
    p = utils.init_parameters()

    # find nearest trained policy to use
    policy_ctrl = trained_ctrls[np.abs(trained_ctrls-ctrl_value).argmin()]

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
    exo_policy_path = utils.init_policypath(env_name_2,run_id,p,suffix="ctrl_"+str(policy_ctrl))
    exo_agent = PPO(exo_state_dim, exo_action_dim, 0, 0, 1, 1, 0.2, True, 0.02)
    exo_agent.load(exo_policy_path)
   
    data_f = open(work_dir+'ResultsData/ExoNN_'+env_name_2+'_'+run_id+suffix+'.csv',"w")

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
            error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
            obs= np.concatenate((state[:2],error,state[3:],exo_l),axis=None)
            done = False
            min_error = target
            time_to_min = 0
            energy = 0
            exoenergy = 0
            max_energy =0
            for t in range(1, epi_len+1):
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
                        str(time_to_min)+','+ str(energy)+','+
                        str(exoenergy)+','+ str(error)+'\n')

    data_f.close()
    env.close()
    return True


def plotWeightVsMinAngleErrorCTRLDes(ctrls,env_name = 'elbow-v0' ,run_id = 'tgt_21'):
    ### Computes MAPE
    if 0.0 not in ctrls:
        ctrls.append(0.0) # for sarc
    ctrls.sort()
    ctrl_type = ['ctrl_'+str(s) for s in ctrls]
    weights = np.array([5])
    env_name2 = env_name+'_'+run_id
    healthy = np.loadtxt(work_dir+"ResultsData/"+env_name2+'.csv', delimiter=',')

    error = np.zeros((len(ctrl_type),len(weights))) # absolute error in reaching the target
    delay = np.zeros((len(ctrl_type),len(weights)))
    energy = np.zeros((len(ctrl_type),len(weights)))
    exoenergy = np.zeros((len(ctrl_type),len(weights)))
    t_stat = np.zeros((len(ctrl_type),len(weights)))
    p_value = np.zeros((len(ctrl_type),len(weights)))
    for j in range(len(ctrl_type)):
        prefix = 'exosarc'
        suffix = ctrl_type[j]
        fileprefix = "ExoNN_"
        if ctrl_type[j]=='ctrl_0.0':
            prefix = 'sarc'
            suffix = ''
            fileprefix=""
        env_name2 = prefix+env_name+'_'+run_id+suffix
        a = np.loadtxt(work_dir+"ResultsData/"+fileprefix+env_name2+'.csv', delimiter=',')
        for i  in range(len(weights)):
            sample_exo = a[np.where(a[:,1]==weights[i]),6]
            sample_hlthy = healthy[np.where(healthy[:,1]==weights[i]),6]
            count = sample_exo.shape[1]
            t_stat[j,i], p_value[j,i] = perform_statistical_test(count,np.average(sample_exo),np.average(sample_hlthy),np.std(sample_exo)*20,np.std(sample_hlthy)*20)
            tmp  = a[np.where(a[:,1]==weights[i]),6]-healthy[np.where(healthy[:,1]==weights[i]),6]
            error[j,i]=np.sqrt(np.average(tmp*tmp))
            # first subtract exo energy from energy to keep only muscle part
            tmp_total  = a[np.where(a[:,1]==weights[i]),4]
            tmp_exo  = a[np.where(a[:,1]==weights[i]),5]
            tmp_mus = np.sqrt(tmp_total*tmp_total-tmp_exo*tmp_exo)
            tmp  = tmp_mus/healthy[np.where(healthy[:,1]==weights[i]),4]-1
            energy[j,i]=np.average(tmp)
            tmp  = a[np.where(a[:,1]==weights[i]),5]
            exoenergy[j,i]=np.average(tmp_exo/tmp_total)
            delay[j,i]=np.average(a[np.where(a[:,1]==weights[i]),3])    
    
    plt.figure(1)
    plt.plot(ctrls,np.average(error,axis=1)/2.1*100,marker='s')
    #print(ctrls,np.average(error,axis=1)/2.1*100)
    plt.xlabel("Control Range Parameter $\mathit{C}$")
    plt.ylabel("RMSD $\delta_{e,h}(\mathit{C})$ (%)")
    plt.title("RMSD of Final Elbow Joint Angle between Exo and Healthy")
    #plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(work_dir+"plots/CTRLDesMinErrorResults_"+run_id+".png")
    plt.close()

    plt.figure(1)
    print(t_stat)
    plt.plot(ctrls,np.average(t_stat,axis=1),marker='s')
    plt.plot(ctrls,1.96*np.ones(len(ctrls),),color='black',linestyle='dashed',linewidth=2)
    #print(ctrls,np.average(error,axis=1)/2.1*100)
    plt.xlabel("Control Range Parameter $\mathit{C}$")
    plt.ylabel("T Statistic")
    plt.title("Two Mean Hypothesis Test at 5% Significance")
    plt.yscale('log')
    #plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(work_dir+"plots/CTRLDesMinErrorT_statistics_"+run_id+".png")
    plt.close()

    plt.figure(2)
    plt.plot(ctrls,np.average(delay,axis=1)*0.01,marker='s')
    plt.title("Average Time Steps to Reach Minimum Flex Angle Error")
    plt.xlabel("Control Range Parameter $\mathit{C}$")
    plt.ylabel("Delay (seconds)")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(work_dir+"plots/CTRLDesMaxDelayResults_"+run_id+".png")
    plt.close()

    plt.figure(3)
    plt.plot(ctrls,np.average(energy[:,:],axis = 1)*100,marker='s')
    plt.title("Percentage Reduction in Muscle Effort Relative to Healthy Condition")
    #plt.legend(loc='upper left')
    plt.xlabel("Control Range Parameter $\mathit{C}$")
    plt.grid()
    plt.ylabel("Relative Muscle Effort Reduction (%)")
    plt.savefig(work_dir+"plots/CTRLDesMuscleEnergyCompareResults_"+run_id+".png")
    plt.close()

    plt.figure(4)
    plt.plot(ctrls,np.average(exoenergy[:,:],axis = 1)*100,marker='s')
    plt.title("Contribution of Exo Assistance to the Total Effort at Various Control Ranges")
    #plt.legend(loc='upper left')
    plt.xlabel("Control Range Parameter $\mathit{C}$")
    plt.ylabel("Relative Exo Effort (%)")
    plt.grid()
    plt.savefig(work_dir+"plots/CTRLDesExoEnergyCompareResults_"+run_id+".png")
    plt.close()

def plotMinAngleErrorCTRLDes(ctrls,env_name = 'elbow-v0' ,run_id = 'tgt_21'):
    # Unused function   
    if 0.0 not in ctrls:
        ctrls.append(0.0) # for sarc
    ctrls.sort()
    ctrl_type = ['ctrl_'+str(s) for s in ctrls]
    weights = np.array([5])
    hlthy_metrics = np.zeros((len(weights),6)) # equivalent to final mean, sd of error state
    env_name2 = env_name+'_'+run_id
    a = np.loadtxt(work_dir+"ResultsData/"+env_name2+'.csv', delimiter=',')
    for i  in range(len(weights)):
        hlthy_metrics[i,0]=np.average(a[np.where(a[:,1]==weights[i]),6]) # E[error]# equivalent to final mean, sd of error state
        tmp = a[np.where(a[:,1]==weights[i]),6]
        hlthy_metrics[i,1] = np.sum(tmp*tmp)/len(tmp) # E(error^2)
        hlthy_metrics[i,2]=np.average(a[np.where(a[:,1]==weights[i]),4]) # E[energy]
        tmp = a[np.where(a[:,1]==weights[i]),4]
        hlthy_metrics[i,3] = np.sum(tmp*tmp)/len(tmp) # E(energy^2)
        hlthy_metrics[i,4]=np.average(a[np.where(a[:,1]==weights[i]),5]) # E[exoenergy]# equivalent to final mean, sd of error state
        tmp = a[np.where(a[:,1]==weights[i]),5]
        hlthy_metrics[i,5] = np.sum(tmp*tmp)/len(tmp) # E(exoenergy^2)

    error = np.zeros((len(ctrl_type),len(weights))) # absolute error in reaching the target
    delay = np.zeros((len(ctrl_type),len(weights)))
    energy = np.zeros((len(ctrl_type),len(weights)))
    exoenergy = np.zeros((len(ctrl_type),len(weights)))
    for j in range(len(ctrl_type)):
        prefix = 'exosarc'
        suffix = ctrl_type[j]
        fileprefix = "ExoNN_"
        if ctrl_type[j]=='ctrl_0.0':
            prefix = 'sarc'
            suffix = ''
            fileprefix=""
        env_name2 = prefix+env_name+'_'+run_id+suffix
        a = np.loadtxt(work_dir+"ResultsData/"+fileprefix+env_name2+'.csv', delimiter=',')
        for i  in range(len(weights)):
            tmp  = a[np.where(a[:,1]==weights[i]),6]
            tmpss, tmpav =  np.sum(tmp*tmp)/len(tmp), np.average(tmp)
            errvar = tmpss-2*tmpav*hlthy_metrics[i,0]+hlthy_metrics[i,1]
            v = np.average(tmp)-hlthy_metrics[i,0]
            error[j,i]=abs(v)
            tmp  = a[np.where(a[:,1]==weights[i]),4]
            tmpss, tmpav =  np.sum(tmp*tmp)/len(tmp), np.average(tmp)
            energyvar = tmpss-2*tmpav*hlthy_metrics[i,2]+hlthy_metrics[i,3]
            energy[j,i]=np.sqrt(energyvar)
            tmp  = a[np.where(a[:,1]==weights[i]),5]
            tmpss, tmpav =  np.sum(tmp*tmp)/len(tmp), np.average(tmp)
            exoenergyvar = tmpss-2*tmpav*hlthy_metrics[i,4]+hlthy_metrics[i,5]
            exoenergy[j,i]=np.sqrt(exoenergyvar)
            delay[j,i]=np.average(a[np.where(a[:,1]==weights[i]),3])    
    
    plt.figure(1)
    plt.scatter(ctrls,np.average(error,axis=1))
    plt.xlabel("Control Range [-x, + x]")
    plt.ylabel("Target Angle Error (Rad)")
    plt.title("Achieved Target Angle Error Relative to Healthy Model")
    plt.legend(loc='upper left')
    plt.savefig(work_dir+"plots/CTRLDesMinErrorResults_"+run_id+".png")
    plt.close()
    plt.figure(2)
    plt.plot(ctrls,np.average(delay,axis=1))
    plt.xlabel("Control Range [-x, + x]")
    plt.ylabel("Average Time Steps to Reach Minimum Flex Angle Error")
    plt.legend(loc='lower right')
    plt.savefig(work_dir+"plots/CTRLDesMaxDelayResults_"+run_id+".png")
    plt.close()
    plt.figure(3)
    plt.scatter(ctrls,np.average(exoenergy[:,:],axis = 1),label='Exo Actuation')
    plt.scatter(ctrls,np.average(energy[:,:],axis = 1),label='Muscle Activation')
    plt.title("Energy Comsumption RMSE Relative to Healthy Model")
    plt.legend(loc='upper left')
    plt.xlabel("Control Range [-x, + x]")
    plt.ylabel("Energy RMSE")
    plt.savefig(work_dir+"plots/CTRLDesEnergyCompareResults_"+run_id+".png")
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

def perform_statistical_test(cnt,m1, m2, s1, s2):
    n1 = cnt
    n2 = cnt
    # this will also be two sided so for 5% significance use 0.025
    stdev = np.sqrt(s1**2/n1+s2**2/n2)
    mean_diff = abs(m1 - m2)
    t_stat = abs(mean_diff/stdev)
    # p-value for 2-sided test
    p_value = 2*(1 - ttest.cdf((t_stat), n1-1))
    return t_stat,p_value


epi_len = 200
#EvaluateAngleWithoutExo(epi_len,sarc=False)
#EvaluateAngleWithoutExo(epi_len,sarc=True)
ctrls = [11.5]
trained_ctrls =np.array([20.0])
registered = False
#for ctrl in ctrls:
#    registered = EvaluateAngleWithExo(ctrl,trained_ctrls,epi_len,registered)
ctrls = [2.5,5.0,7.5,10.0,11.0,12.5,15.0,17.5,20.0,22.5,25.0]
plotWeightVsMinAngleErrorCTRLDes(ctrls)
#plotMinAngleErrorCTRLDes(ctrls)
#plotErrorMinCTRLDes(ctrls)






