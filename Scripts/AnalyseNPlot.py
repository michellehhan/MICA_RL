#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:19:19 2023

@author: goyal
"""   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scs
work_dir = 'CodeColab/LE/Squat2Jump/'

def test(a,b, significance=0.05):
    m_a = np.average(a)
    m_b = np.average(b)
    std_a = np.std(a)   
    std_b = np.std(b)
    t = (m_a-m_b)/(np.sqrt(std_a*std_a/len(a)+std_b*std_b/len(b)))
    dof = min(len(a),len(b))-1
    print(t,dof)
    p_value = (1-scs.stats.t.cdf(abs(t),dof))*2
    return m_a, std_a, m_b, std_b, t, p_value, p_value<significance


def plotMaxHeightCTRLDes(env_name = 'myoLegReachFixed-v0' ,run_id = str(4)):
    ctrl_range = ['1000.0','900.0','800.0','700.0','600.0','500.0']
    ctrl_type = ['ctrl_'+s for s in ctrl_range]
    ctrl_type.extend(['sarc','hlthy'])
    height = np.zeros((len(ctrl_type),1))
    delay = np.zeros((len(ctrl_type),1))
    energy = np.zeros((len(ctrl_type),1))
    exoenergy = np.zeros((len(ctrl_type),1))

    for j in range(len(ctrl_type)):
        prefix = 'exosarc'
        suffix = '_'+ctrl_type[j]
        if ctrl_type[j]=='hlthy':
            prefix = ''
            suffix = ''
        if ctrl_type[j]=='sarc':
            prefix = 'sarc'
            suffix = ''
        env_name2 = prefix+env_name+'_'+run_id+suffix
        a = np.loadtxt(work_dir+'ResultsData/'+env_name2+'.csv', delimiter=',')
        height[j]=np.average(a[:,0])
        delay[j]=np.average(a[:,1])
        energy[j]=np.average(a[:,2])
        exoenergy[j]=np.average(a[:,3])
    
    plt.figure(1)
    plt.scatter(ctrl_type,height)
    plt.xlabel("Control Scenario")
    plt.ylabel("Maximum Jump Height (m)")
    plt.savefig(work_dir+"plots/CTRLDesMaxHeightResults_"+run_id+".png")

    plt.figure(2)
    plt.scatter(ctrl_type,delay)
    plt.xlabel("Control Scenario")
    plt.ylabel("Time to Reach Max Height ")
    plt.savefig(work_dir+"plots/CTRLDesDelayResults_"+run_id+".png")

    plt.figure(3)
    plt.scatter(ctrl_type,energy)
    plt.scatter(ctrl_type,exoenergy)
    plt.xlabel("Control Scenario")
    plt.ylabel("Energy Comsumption ")
    plt.savefig(work_dir+"plots/CTRLDesEnergyResults_"+run_id+".png")
    plt.close()


def plotExoActionsCTRLDes(env_name = 'elbow-v0' ,run_id = 'random_wt'):
    ctrl_type = ['ctrl_10','ctrl_50']
    weights = np.array([1,2,3,4,5])
    actions = np.zeros((1,100))
    plt.figure(1)
    for j in range(len(ctrl_type)):
        prefix = 'exosarc'
        suffix = '_'+ctrl_type[j]
        if ctrl_type[j]=='no exo':
            prefix = 'norm'
            suffix = ''
        env_name2 = prefix+env_name+'_'+run_id+suffix
        a = np.loadtxt('CodeColab/ResultsData/UE/ElbowFlex/Exp2/'+env_name2+'Actions.csv', delimiter=' ')
        #print(a[2,:].shape)
        plt.figure(1)
        for i  in range(len(weights)):
            actions=a[i,:]
            plt.plot(a[i,:],label=ctrl_type[j]+'_wt_'+str(i+1))
    plt.figure(1)
    plt.xlabel("Episode Steps")
    plt.ylabel("Exo Action")
    plt.legend(loc='upper right')
    plt.savefig("CodeColab/plots/UE/ElbowFlex/Exp2/CTRLDesExoActionsResults_"+run_id+".png")
    #plt.close()
    plt.close()


def plotAvgFlexAngleExoCtrlRange(env_name = 'elbow-v0' ,run_id = '15'):
    env_type = ['sarc']
    ctrl_range = [0,10,20,30,40,50,60,70,80]
    bins = np.linspace(0.0, 2.5, 100)
    d = np.zeros((1000,len(ctrl_range)))
    angles = np.zeros(len(ctrl_range))
    energy = np.zeros(len(ctrl_range))
    plt.figure()
    for i in range(len(ctrl_range)):
        for e in env_type:
            if ctrl_range[i] ==0:
                env_name_2 = e+env_name
                if e == 'norm':
                    env_name_2 = env_name
                a = np.loadtxt('CodeColab/ResultsData/UE/ElbowFlex/'+env_name_2+'_'+run_id+ '.csv', delimiter=',')
            else:
                env_name_2 = 'exo'+e+env_name
                if e == 'norm':
                    env_name_2 = 'exo'+env_name
                a = np.loadtxt('CodeColab/ResultsData/UE/ElbowFlex/'+env_name_2+'_'+run_id+'_ctrl_'+str(ctrl_range[i])+'.csv', delimiter=',')               
            angles[i]=np.average(a[:,0])
            energy[i]=np.average(a[:,2])
    #print(heights)
    #sprint(energy)
    plt.scatter(ctrl_range,angles)        
    plt.ylabel("Maximum Angle of Flex")
    plt.xlabel("Exo Control Range")
    plt.savefig("CodeColab/plots/UE/ElbowFlex/AvgMaxFlexResults_"+run_id+"_ctrlDesign.png")
    plt.close()

def plotMscActData(env_name = 'elbow-v0',run_id = '15'):
    env_type = ['exosarc','sarc','exo','normal']
    bins = np.linspace(0, 1000, 100)
    for e in env_type:
        env_name_2 = e+env_name
        if e == 'normal':
            env_name_2 = 'norm'+env_name
        a = np.loadtxt('CodeColab/ResultsData/UE/ElbowFlex/'+env_name_2+'_'+run_id+'.csv', delimiter=',')
        print(len(a))
        plt.hist(a[:,2], bins, alpha=0.5, label=e)

    plt.legend(loc='upper right')
    plt.xlabel("Muscle Energy Consumption Rates")
    plt.ylabel("Frequency")
    plt.savefig("CodeColab/plots/UE/ElbowFlex/MscEnergyRatesResults_"+run_id+".png")
    plt.close()

def plotReachDelayData(env_name = 'elbow-v0',run_id = '15'):
    env_type = ['exosarc','sarc','exo','normal']
    bins = np.linspace(25, 75, 50)
    for e in env_type:
        env_name_2 = e+env_name
        if e == 'normal':
            env_name_2 = 'norm'+env_name
        a = np.loadtxt('CodeColab/ResultsData/UE/ElbowFlex/'+env_name_2+'_'+run_id+'.csv', delimiter=',')
        print(len(a))
        plt.hist(a[:,1], bins, alpha=0.5, label=e)

    plt.legend(loc='upper right')
    plt.xlabel("Reach Delay (sec)")
    plt.ylabel("Frequency")
    plt.savefig("CodeColab/plots/UE/ElbowFlex/ReachDelayResults_"+run_id+".png")
    plt.close()

def plotExoCostData(env_name = 'elbow-v0',run_id = '15'):
    env_type = ['sarc','normal']
    bins = np.linspace(0, 0.5, 50)
    for e in env_type:
        env_name_2 = 'exo'+e+env_name
        if e == 'normal':
            env_name_2 = 'exo'+env_name
        a = np.loadtxt('CodeColab/ResultsData/UE/ElbowFlex/'+env_name_2+'_'+run_id+'.csv', delimiter=',')
        print(len(a))
        plt.hist(a[:,3], bins, alpha=0.5, label=e)

    plt.legend(loc='upper right')
    plt.xlabel("Exo Cost")
    plt.ylabel("Frequency")
    plt.savefig("CodeColab/plots/UE/ElbowFlex/ExoCostResults_"+run_id+".png")
    plt.close()
 
#testAngleData(['sarc','normal'])  
#testAngleData(['exo','normal'])  
#testAngleData(['exosarc','normal'])  

#plotFlexAngleData() 
#plotAvgFlexAngleExoCtrlRange() 
#plotFlexAngleData()
plotMaxHeightCTRLDes()
#plotExoActionsCTRLDes()
