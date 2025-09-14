import gym
import os
import numpy as np
import math
from datetime import datetime
from myosuite.envs.env_variants import register_env_variant


path = 'UE/ElbowFixTarget/'
env_name='myoElbowPose1D6MExoRandom-v0'

def init_logpath(env_name_2,run_id,p, suffix=""):
    ###################### logging ######################
    if suffix!="":
        suffix = "_"+suffix
    log_dir = p['work_dir']+'logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_f_name = log_dir + 'PPO_' + env_name_2 + "_log_" + run_id +suffix+ ".csv"
    print("logging at : " + log_f_name)
    return log_f_name

def init_policypath(env_name_2,run_id,p, suffix=""):
    ################### checkpointing ###################
    if suffix!="":
        suffix = "_"+suffix
    directory = p['work_dir']+'Policies/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = directory + 'PPO_' + env_name_2 + "_"+run_id+suffix+ ".pth" 
    print("policy path : " + checkpoint_path)
    return checkpoint_path

def get_healthypolicypath(p,run_id):
    directory = p['work_dir']+'Policies/'
    healthy_policy_path = directory + "PPO_{}_{}.pth".format(env_name, run_id)
    print("Load Healthy Policy from : " + healthy_policy_path)
    return healthy_policy_path

def init_parameters(train_steps = 6e5): # for holding a pose.. make find action std dev to zero; USe 600K steps to guarantee this
    p = {}
    p['model_dir'] = 'CodeColab/MyoAssist/Model/'
    p['work_dir'] = 'CodeColab/'+path
    p['new_model_nm']= 'elbow-v0'
    ####### initialize environment hyperparameters ######
    p['has_continuous_action_space'] = True
    p['max_ep_len'] = 50                    # max timesteps in one episode
    p['max_training_timesteps'] = int(train_steps)   # break training loop if timeteps > max_training_timesteps

    p['print_freq'] = int(p['max_training_timesteps'] / 100)     # print avg reward in the interval (in num timesteps)
    p['log_freq'] = int(p['max_training_timesteps'] / 1000 )      # log avg reward in the interval (in num timesteps)
    p['save_model_freq'] = int(p['max_training_timesteps']/6)      # save model frequency (in num timesteps)

    p['action_std'] = 0.6                    # starting std for action distribution (Multivariate Normal)
    p['action_std_decay_rate'] = 0.1        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    p['min_action_std'] = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    p['action_std_decay_freq'] = int(p['max_training_timesteps']/6)

    ################ PPO hyperparameters ################
    p['update_timestep'] = p['max_ep_len'] * 4      # update policy every n timesteps
    p['K_epochs'] = 20               # update policy for K epochs
    p['eps_clip'] = 0.2              # clip parameter for PPO
    p['gamma'] = 0.99                # discount factor

    p['lr_actor'] = 0.0003       # learning rate for actor network
    p['lr_critic'] = 0.001       # learning rate for critic network

    p['random_seed'] = 0         # set random seed if required (0 = no random seed)
    return p

def print_parameters(p):
    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", p['max_training_timesteps'])
    print("max timesteps per episode : ", p['max_ep_len'])
    print("model saving frequency : " + str(p['save_model_freq']) + " timesteps")
    print("log frequency : " + str(p['log_freq']) + " timesteps")
    print("printing average reward over episodes in last : " + str(p['print_freq']) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    if p['has_continuous_action_space']:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", p['action_std'])
        print("decay rate of std of action distribution : ", p['action_std_decay_rate'])
        print("minimum std of action distribution : ", p['min_action_std'])
        print("decay frequency of std of action distribution : " + str(p['action_std_decay_freq']) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(p['update_timestep']) + " timesteps") 
    print("PPO K epochs : ", p['K_epochs'])
    print("PPO epsilon clip : ", p['eps_clip'])
    print("discount factor (gamma) : ", p['gamma'])
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", p['lr_actor'])
    print("optimizer learning rate critic : ", p['lr_critic'])
    print("============================================================================================")

def show_video(video_path, video_width = 500):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

def set_init_pose(env, key_name='stand'):
    env.sim.data.qpos = env.sim.model.keyframe(key_name).qpos
    env.sim.data.qvel = env.sim.model.keyframe(key_name).qvel
    env.forward()

def get_target_pose(env,target):
    set_init_pose(env,key_name=target)
    #pelvis_height = env.sim.data.joint('pelvis_ty').qpos[0].copy()
    return env.sim.data.qpos.copy()


############# reward function #################
def get_reward(old_err,new_err):
    oe, ne = abs(old_err),  abs(new_err)
    # First reach near target and then get additional reward for staying there
    r= 10*math.exp(-10*abs(ne))-5 #+ (10 * (oe > ne)+ oe-ne)*(oe<0.1)
    return r

def get_reward_err_delta(old_err,new_err):
    oe, ne = abs(old_err),  abs(new_err)
    return 10 * (oe >= ne) -10 * (oe < ne)+ oe-ne

def get_exo_state(env):
    a = np.zeros((4,))
    a[0] = env.sim.data.actuator('Exo_R').length.item(0)
    a[1] = env.sim.data.actuator('Exo_L').length.item(0)
    a[2] = env.sim.data.actuator('Exo_R').velocity.item(0)
    a[3] = env.sim.data.actuator('Exo_L').velocity.item(0)
    return a

def get_env_prefix(exo,sarc):
    prefix = ""
    if exo:
        prefix +='exo'
    if sarc:
        prefix += 'sarc'
    return prefix

def register_env(p,exo,sarc):
    env_name_2 = get_env_prefix(exo,sarc)+p['new_model_nm']
    if exo and sarc:
        register_env_variant( env_id= env_name, variants={'normalize_act':False, 'muscle_condition':'sarcopenia'}, variant_id= env_name_2, silent=False )
    elif exo and not(sarc):
        register_env_variant( env_id= env_name, variants={'normalize_act':False}, variant_id= env_name_2,silent=False )
    elif not(exo) and sarc:
        register_env_variant( env_id= env_name, variants={'normalize_act':False, 'muscle_condition':'sarcopenia'}, variant_id= env_name_2,silent=False )
    else:
        register_env_variant( env_id= env_name, variants={'normalize_act':False}, variant_id= env_name_2,silent=False )
    return env_name_2

def get_current_pose(env):
    return env.sim.data.qpos

def get_pose_err(env,target,ids):
    #print(env.env.sim.data.qpos[ids])
    #print(target)
    err= env.sim.data.qpos[ids]-target
    return err.copy()


def printReward(reward,episodes,i_episode, time_step):
    # print average reward till last episode
    print_avg_reward = reward / episodes
    print_avg_reward = round(print_avg_reward, 2)
    print("Episode : {} \t Timestep : {} \t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
    return 0, 0

def printModelSaved(checkpoint_path, start_time):
    print("--------------------------------------------------------------------------------------------")
    print("saving model at : " + checkpoint_path)
    print("model saved")
    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
    print("--------------------------------------------------------------------------------------------")

def printLogs(log_f, reward,episodes,i_episode, time_step):
    # log average reward till last episode
    log_avg_reward = reward / episodes
    log_avg_reward = round(log_avg_reward, 4)

    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
    log_f.flush()
    return 0, 0

def scaleAction(act2,ctrl_lo,ctrl_hi):
    return (act2)* (ctrl_hi-ctrl_lo)/2+  (ctrl_hi+ctrl_lo)/2

def init_plot_data(env,exo, p):
    num_state = 45
    # Actuators: 22 muscles, 2 exo
    num_act = 22  
    if exo:
        num_act +=2 
    plot_data = {'mus_act': None, "jnt_state": None}
    for idx in range(num_act):
        plot_data['mus_act'][env.sim.data.actuator(idx).name]=np.zeros((p['max_ep_len'],))
    for idx in range(num_state):
        plot_data['jnt_state'][env.sim.data.jnt(idx).name]=np.zeros((p['max_ep_len'],))
    plot_data['knee_r_torque']=np.zeros((p['max_ep_len'],))
    plot_data['knee_l_torque']=np.zeros((p['max_ep_len'],))   
    return plot_data

def update_plot_data(plot_data, env,exo, p, time):
    for mus in range(num_act):
        plot_data['mus_act'][env.sim.data.actuator(idx).name][time]=np.zeros((p['max_ep_len'],))
    for idx in range(num_state):
        plot_data[env.sim.data.jnt(idx).name]=np.zeros((p['max_ep_len'],))
    plot_data['knee_r_torque']=np.zeros((p['max_ep_len'],))
    plot_data['knee_l_torque']=np.zeros((p['max_ep_len'],))   
    return plot_data

def get_plot_data(env,exo,error):
    num_state = 1 # join angle pos
    # Actuators: 1 exo and 6 muscles 
    num_act = 7  
    temp_headers = [None]*num_act
    temp_stim = np.zeros(num_act,)
    temp_activation = np.zeros(num_act,)
    plot_data = {}
    for idx in range(num_act):
        temp_headers[idx] = env.sim.data.actuator(idx).name 
        temp_stim[idx] = env.sim.data.actuator(idx).ctrl[0].copy()
        if temp_headers[idx].startswith('Exo'):
            temp_activation[idx] =  env.sim.data.actuator('Exo').length.item(0)
        else:
            temp_activation[idx] = env.sim.data.act[idx-1].copy()
    plot_data['act_headers'] = temp_headers
    plot_data['act_ctrl'] = temp_stim
    plot_data['activation'] = temp_activation
    
    temp_headers = [None]*num_state
    temp_stim = np.zeros(num_state,)
    for idx in range(num_state):
        temp_headers[idx] = env.sim.data.jnt(idx).name 
        temp_stim[idx] = env.sim.data.jnt(idx).qpos[0].copy()  
    plot_data['joint_headers'] = temp_headers
    plot_data['joint_qpos'] = temp_stim
    plot_data['r_elbow_torque']=env.sim.data.joint('r_elbow_flex').qfrc_constraint[0].copy()+env.sim.data.joint('r_elbow_flex').qfrc_smooth[0].copy()
    plot_data['angle_error']=error

    return plot_data

def unpack_plot_dict(data_dict):
    unpacked_labels = data_dict[0]['plt_dict']['act_headers']
    unpacked_joint_labels = data_dict[0]['plt_dict']['joint_headers']
    unpacked_stim = np.zeros((len(data_dict), len(data_dict[0]['plt_dict']['act_headers'])))
    unpacked_activ = np.zeros((len(data_dict), len(data_dict[0]['plt_dict']['act_headers'])))
    unpacked_joint = np.zeros((len(data_dict),len(data_dict[0]['plt_dict']['joint_headers'])))
    unpacked_torque_r = np.zeros(len(data_dict),)
    unpacked_error_r = np.zeros(len(data_dict),)
    
    for data_idx in range(len(data_dict)):
        for item_idx in range( len(data_dict[0]['plt_dict']['act_ctrl']) ):
            unpacked_stim[data_idx, item_idx] = data_dict[data_idx]['plt_dict']['act_ctrl'][item_idx]
            unpacked_activ[data_idx, item_idx] = data_dict[data_idx]['plt_dict']['activation'][item_idx]
        for item_idx in range( len(data_dict[0]['plt_dict']['joint_qpos']) ):
            unpacked_joint[data_idx, item_idx] = data_dict[data_idx]['plt_dict']['joint_qpos'][item_idx]
        unpacked_torque_r[data_idx] = data_dict[data_idx]['plt_dict']['r_elbow_torque']
        unpacked_error_r[data_idx] = data_dict[data_idx]['plt_dict']['angle_error']
    
    unpacked_dict = {'act_headers': unpacked_labels, 
                     'act_ctrl': unpacked_stim, 
                     'activation': unpacked_activ, 
                     'joint_header': unpacked_joint_labels,
                     'joint_angles': unpacked_joint, 
                     'r_elbow_torque': unpacked_torque_r,
                     'angle_error': unpacked_error_r}
    
    return unpacked_dict