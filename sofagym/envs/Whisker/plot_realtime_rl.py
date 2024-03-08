import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('fivethirtyeight')
import pandas as pd
import os
import csv

fieldnames = ["Sim_step", "lamda_xx", "lamda_yy", "lamda_zz","lamda_yz", "lamda_xz", "lamda_xy"]
count = 0


def rename_csv(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"File renamed from {old_name} to {new_name}")
    except FileNotFoundError:
        print(f"File {old_name} not found.")
    except FileExistsError:
        print(f"File {new_name} already exists.")

old_filename = 'strain_data.csv'
new_filename = 'previous_strain_data.csv'

# Create the figure and axis outside the animate function
fig, ax = plt.subplots(nrows=1, ncols=2,sharex=False,sharey=False,figsize=(10, 5))
measured_ele = [1785,1392,1336,1561]
ele = 1785
def animate(i):
    data = pd.read_csv("strain_data_rl/strain_" + str(ele)+".csv")  
    x = data['Sim_step']
    y1 = data['lamda_xx']
    y2 = data['lamda_yy']
    y3 = data['lamda_zz']
    y4 = data['lamda_yz']
    y5 = data['lamda_xz']
    y6 = data['lamda_xy']

    # Only use the most recent 100 data points
    start_idx = max(0, len(x) - 50)
    end_idx = len(x)
    x = x[start_idx:end_idx]
    y1 = y1[start_idx:end_idx]
    y2 = y2[start_idx:end_idx]
    y3 = y3[start_idx:end_idx]
    y4 = y4[start_idx:end_idx]
    y5 = y5[start_idx:end_idx]
    y6 = y6[start_idx:end_idx]

    ax[0].clear()
    ax[1].clear()


    # Set x-axis limits to keep the scale constant
    
    # ax.plot(x, y1, label='lamda_xx')
    # ax.plot(x, y2, label='lamda_yy')
    # ax.plot(x, y3, label='lamda_zz')
    # ax.plot(x, y4, label='lamda_yz')
    ax[0].plot(x, y5, label='lamda_xz')
    # ax.plot(x, y6, label='lamda_xy')
    
    # Plot ground truth data:
    data = pd.read_csv("strain_groundtruth/strain_" + str(ele)+".csv")
    x = data['Sim_step']
    y1_t = data['lamda_xx']
    y2_t = data['lamda_yy']
    y3_t = data['lamda_zz']
    y4_t = data['lamda_yz']
    y5_t = data['lamda_xz']
    y6_t = data['lamda_xy']

    ax[0].plot(x, y5_t, label='lamda_xz_truth',color = 'black',linestyle='dashed')
    ax[0].legend(loc='upper left')
    ax[0].set_xlim(x.min(), x.max())
    # ax[0].set_ylim(-0.000002,0.000005)

    score = pd.read_csv("strain_data_rl/reward_cost_record.csv")
    time = score['Sim_step']
    cost = score['cost']
    reward = score['reward']
    # ax[1].plot(time,cost,label='cost')
    ax[1].plot(time,reward,label='reward')
    ax[1].legend(loc='upper left')
    # ax[1].set_xlim(time.min(), time.max())


ani = FuncAnimation(fig, animate, frames=10, interval=10)

plt.tight_layout()
plt.show()



