# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:49:44 2023

@author: athan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:28:20 2023

@author: athan
"""
from psychopy import visual, core, logging, clock
from psychopy.hardware import keyboard
from psychopy import monitors
from psychopy import gui, parallel
import psychopy.event
import numpy as np
import random
import asyncio
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import random
import psychopy.event
import numpy as np
import math
import serial
import threading
import time
import gc
import matplotlib.pyplot as plt
from psychopy.iohub import launchHubServer
import sched
import pandas as pd



#PORT HANDLING UNCOMMENT
#port = parallel.ParallelPort(address=0x3FF0)

#Left Trigger = 3, Right==4, their ends are 6 and 8 respectively

#Text fields-Self Explanatory
myDlg = gui.Dlg(title="SSVEP Experiment")
myDlg.addField('Participant Number')
myDlg.addField('ORDER_CONDITION):')

ok_data = myDlg.show() 


# CREATION OF CONDITIONS

nums = [num for num in range(180)]

trial_num = 1
data = []

# Shuffle the list randomly
random.shuffle(nums)

# Divide the list into 4 equal sublists
num_per_list = len(nums) // 4
sublists = [nums[i:i+num_per_list] for i in range(0, len(nums), num_per_list)]

Left_Low = sublists[0]
Left_High = sublists[1]
Right_Low = sublists[2]
Right_High = sublists[3]



def ATTEND(TRIAL):
    # noob_probability=[0,1] #CHANGE THIS
    global frequency, RIGHT_CUE, LEFT_CUE, CUE, frequency_ANT, frequency_MAIN
    #LEFT_CUE=visual.TextStim(win, text="<<<",pos=(-150,0))
    #RIGHT_CUE=visual.TextStim(win, text=">>>", pos=(150,0))
    LEFT_CUE=visual.ImageStim(win,image='LookLeft.png',pos=(0,150), size=128)
    RIGHT_CUE=visual.ImageStim(win,image='LookRight.png',pos=(0,150), size=128)
    #FIFTY_PER_CENT=random.choice(noob_probability)
    if TRIAL in Left_Low:
            LEFT_CUE.autoDraw=True
            RIGHT_CUE.autoDraw=False
            #RIGHT_CUE=visual.TextStim(win, text="", color='white', pos=(80,0),opacity=0)
            port.setData(0)
            time.sleep(0.04)
            port.setData(4)
            time.sleep(0.04)
            port.setData(0)
            CUE=0
            frequency_MAIN=3.75
            frequency_ANT=4.8
    if TRIAL in Left_High:
            LEFT_CUE.autoDraw=True
            RIGHT_CUE.autoDraw=False
            #RIGHT_CUE=visual.TextStim(win, text="", color='white', pos=(80,0),opacity=0)
            port.setData(0)
            time.sleep(0.04)
            port.setData(6)
            time.sleep(0.04)
            port.setData(0)
            CUE=0
            frequency_MAIN=4.8
            frequency_ANT=3.75
    if  TRIAL in Right_Low:
            RIGHT_CUE.autoDraw=True
            LEFT_CUE.autoDraw=False
            #LEFT_CUE=visual.TextStim(win, text="", color='white',pos=(-80,0),opacity=0)
            port.setData(0)
            time.sleep(0.04)
            port.setData(8)
            port.setData(0)
            time.sleep(0.04)
            CUE=1
            frequency_MAIN=3.75
            frequency_ANT=4.8
    if TRIAL in Right_High:
            RIGHT_CUE.autoDraw=True
            LEFT_CUE.autoDraw=False
            #LEFT_CUE=visual.TextStim(win, text="", color='white',pos=(-80,0),opacity=0)
            port.setData(0)
            time.sleep(0.04)
            port.setData(9)
            port.setData(0)
            time.sleep(0.04)
            CUE=1
            frequency_MAIN=4.8
            frequency_ANT=3.75
    return frequency_ANT, frequency_MAIN,CUE


# Open a window
win = visual.Window(size=(1920,1080), units="pix", color='black', fullscr=True)

Instructions=visual.TextStim(win, font='Songti SC', text="1. 总是看十字架的中心。, 2.当你直视十字架时，用你的注意力/侧视来看箭头指向的地方。 3.如果箭头指向右侧，右侧出现X，则按一次键盘上的字母C。, 4.如果箭头指向右侧并且出现X，请按字母m一次。 5.如果箭头不指向该区域，请完全忽略该区域，不要按该区域的任何内容。仅关注箭头指向的区域。, 6. 当您按下 C 或 M 按钮时，请尽可能快和准确。如果你错过了其中一些，那就尽力而为吧。, 7.每6分钟休息一次，让您休息约1至2分钟。 按空格键继续 Press SPACE ")
Instructions.draw()


Break=visual.TextStim(win, font='Songti SC', text="BREAK 1 Minute. Press SPACE to continue ")

win.flip()
event.waitKeys()

Fixation_Cross = visual.GratingStim(win, tex='fixation_cross.png', size=100, pos=(0,0), opacity=1)
Fixation_Cross.autoDraw=True

global letters

letters= ['A', 'B','Z', 'X', 'O', 'I', '3', '5', 'K']


gaborRight= visual.Rect(win, color='white', size=300,  opacity=1, pos=(525, -250), autoDraw=False)

#################################################################################

gaborLeft= visual.Rect(win, color='white', size=300,  opacity=1, pos=(-525, -250), autoDraw=False)

# 5-7, 6-8, 6-8.571, 10-12
k=0
x=0

def draw_SSVEP(timer,frequency_MAIN,frequency_ANT,CUE):
    if CUE==0:
        if(math.sin((timer.getTime())*2*math.pi*frequency_MAIN))>0:
            gaborLeft.draw()
        if(math.sin((timer.getTime())*2*math.pi*frequency_ANT))>0:
            gaborRight.draw()
    if CUE==1:
        if(math.sin((timer.getTime())*2*math.pi*frequency_MAIN))>0:
            gaborRight.draw()
        if(math.sin((timer.getTime())*2*math.pi*frequency_ANT))>0:
            gaborLeft.draw()
    return CUE


keys = psychopy.event.getKeys(keyList=['space'], timeStamped=(True)) 

Break=visual.TextStim(win, font='Songti SC', text="BREAK 1 Minute. Press SPACE to continue ")
Break.pos=(0,-300)
Experiment_start=clock.Clock()

for TRIAL in range(0,180):         ## CHANGE THIS ORIGINAL 180
    if TRIAL==50:
        Break.draw()
        win.flip()
        core.wait(0.5)
        event.waitKeys()
    if TRIAL==100:
        Break.draw()
        win.flip()
        core.wait(0.5)
        event.waitKeys()
    if TRIAL==125:
        Break.draw()
        win.flip()
        core.wait(0.5)
        event.waitKeys()
    if TRIAL==150:
        Break.draw()
        win.flip()
        core.wait(0.5)
        event.waitKeys()
    # PLUG IN SOME REFRESH FOR THE CUE
    ATTEND(TRIAL)
    win.flip()
    core.wait(1.1)
    ss=[]
    x_left=[]
    x_right=[]
    Current_Letter='S'
    Current_Letter_Left='S'
    Current_Letter_Right='S'
    All_Left_X=[]
    All_Right_X=[]
    flag_left=0
    flag_right=0
    newClock = clock.Clock()
    Cooldown_X_Left=clock.Clock()
    Cooldown_X_Right=clock.Clock()
    CD_Left=500
    CD_Right=500
    timer = core.CountdownTimer(10)
    letters_left= ['A', 'B','Z', 'X', 'O', 'I', '3', '5', 'K']
    letters_right=['A', 'B','Z', 'X', 'O', 'I', '3', '5', 'K']
    newClock.reset()
    while timer.getTime()>=0:
        draw_SSVEP(timer,frequency_MAIN, frequency_ANT,CUE)
        if CUE==0:
            keys = psychopy.event.getKeys(keyList=['c', 'm'], timeStamped=newClock)
            if len(keys)>0:
                ss.append(keys)
            if(math.sin((timer.getTime())*2*math.pi*frequency_MAIN))>0:
                  Text_Stim_Left=visual.TextStim(win, color="red", text=Current_Letter_Left, pos=(-525, -250), opacity=1)
                  Text_Stim_Left.size=300
                  Text_Stim_Left.draw()
                  if Text_Stim_Left.text=='X' and flag_left==0:
                      x_left=[newClock.getTime(), "X"]
                      All_Left_X.append(x_left)
                      flag_left=1
                      Cooldown_X_Left.add(-0.35)
                      CD_Left=Cooldown_X_Left.getTime()
                      letters_left= ['A', 'B','Z', 'O', 'I', '3', '5', 'K']
                  if newClock.getTime()>CD_Left:
                          letters_left= ['A', 'B','Z', 'X', 'O', 'I', '3', '5', 'K']
                          flag_left=0
            #keys = psychopy.event.getKeys(keyList=['c', 'm'], timeStamped=(True))
            #if len(keys)>0:
                #ss.append(keys)
                #ss.append(newClock.getTime())
            if(math.sin((timer.getTime())*2*math.pi*frequency_ANT))>0:
                  Text_Stim_Right=visual.TextStim(win, color='red', text=Current_Letter_Right, pos=(525, -250), opacity=1)
                  Text_Stim_Right.size=300
                  Text_Stim_Right.draw()
                  if Text_Stim_Right.text=='X' and flag_right==0:
                      x_right=[newClock.getTime(), "X"]
                      All_Right_X.append(x_right)
                      flag_right=1
                      Cooldown_X_Right.add(-0.35)
                      CD_Right=Cooldown_X_Right.getTime()
                      letters_right= ['A', 'B','Z', 'O', 'I', '3', '5', 'K']
                  if newClock.getTime()>CD_Right:
                          letters_right= ['A','B','Z', 'X', 'O', 'I', '3', '5', 'K']
                          flag_right=0
            if(math.sin((timer.getTime())*2*math.pi*frequency_MAIN))<0:
               Current_Letter_Left=random.choice(letters_left)
            if(math.sin((timer.getTime())*2*math.pi*frequency_ANT))<0:
               Current_Letter_Right=random.choice(letters_right)
        if CUE==1:
            keys = psychopy.event.getKeys(keyList=['c', 'm'], timeStamped=newClock)
            if len(keys)>0:
                ss.append(keys)
            if(math.sin((timer.getTime())*2*math.pi*frequency_ANT))>0:
                  Text_Stim_Left=visual.TextStim(win, color="red", text=Current_Letter_Left, pos=(-525, -250), opacity=1)
                  Text_Stim_Left.size=300
                  Text_Stim_Left.draw()
                  if Text_Stim_Left.text=='X' and flag_left==0:
                      x_left=[newClock.getTime(), "X"]
                      All_Left_X.append(x_left)
                      flag_left=1
                      Cooldown_X_Left.add(-0.35)
                      CD_Left=Cooldown_X_Left.getTime()
                      letters_left= ['A', 'B','Z', 'O', 'I', '3', '5', 'K']
                  if newClock.getTime()>CD_Left:
                          letters_left= ['A', 'B','Z', 'X', 'O', 'I', '3', '5', 'K']
                          flag_left=0
            if(math.sin((timer.getTime())*2*math.pi*frequency_MAIN))>0:
                  Text_Stim_Right=visual.TextStim(win, color='red', text=Current_Letter_Right, pos=(525, -250), opacity=1)
                  Text_Stim_Right.size=300
                  Text_Stim_Right.draw()
                  if Text_Stim_Right.text=='X' and flag_right==0:
                      x_right=[newClock.getTime(), "X"]
                      All_Right_X.append(x_right)
                      flag_right=1
                      Cooldown_X_Right.add(-0.35)
                      CD_Right=Cooldown_X_Right.getTime()
                      letters_right= ['A', 'B','Z', 'O', 'I', '3', '5', 'K']
                  if newClock.getTime()>CD_Right:
                          letters_right= ['A', 'B','Z', 'X', 'O', 'I', '3', '5', 'K']
                          flag_right=0
            if(math.sin((timer.getTime())*2*math.pi*frequency_ANT))<0:
               Current_Letter_Left=random.choice(letters_left)
            if(math.sin((timer.getTime())*2*math.pi*frequency_MAIN))<0:
               Current_Letter_Right=random.choice(letters_right)
        win.flip()
    RIGHT_CUE.autoDraw=False
    LEFT_CUE.autoDraw=False
    data.append([trial_num, CUE, newClock.getTime(), ss, All_Left_X, All_Right_X,  frequency_MAIN, Experiment_start.getTime()])
    trial_num += 1
    gc.collect() #Consider uncommenting
    core.wait(0.1)


df2 = pd.DataFrame(data, columns=['Trial Number', 'CUE', 'Trial_Clock', 'TimestampKey', 'X Left', 'X Right',  'frequency_MAIN', 'Experiment Start Time'])
df2.to_csv(str(ok_data[0])+str(ok_data[1])+'.csv')   


win.close()
core.quit()
