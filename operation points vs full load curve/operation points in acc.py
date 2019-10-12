import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

root_dir = r'C:\Users\lvhui\Desktop\operation points\\'
file_list = os.listdir(root_dir)
file_list = [file_list[i] for i in range(0, len(file_list)) if '.csv' in file_list[i]]
dbc = {'torque': 'EnToqActuExtdRngHSC1', 'enspd': 'EnSpdHSC1', 'pedal': 'AccelActuPosHSC1',
       'veh_dri': 'VehSpdAvgDrvnHSC1', 'veh_non': 'VehSpdAvgNonDrvnHSC1'}
cut_time = 7
result_dict = {}
fre = 20

for file_name in file_list:
    full_load_curve = pd.read_csv(r'C:\Users\lvhui\Desktop\full load curve.csv')
    data = pd.read_csv(os.path.join(root_dir, file_name), encoding='GB18030')
    torque_data = data[dbc['torque']].tolist()
    enspd_data = data[dbc['enspd']].tolist()
    pedal_data = data[dbc['pedal']].tolist()
    vehdri_data = data[dbc['veh_dri']].tolist()
    vehnon_data = data[dbc['veh_non']].tolist()
    start_index = []
    end_index = []
    for i, j in groupby(enumerate(np.array(np.where(data[dbc['pedal']] > 99))[0]), lambda x: x[1] - x[0]):
        segment_index = [index for group,index in j]
        if len(segment_index) > fre*cut_time:
            start_index.append(segment_index[0])
            end_index.append(segment_index[-1])
        else:
            continue

color_dict = {1: 'r', 2: 'g', 3: 'b'}
for i in range(0, len(start_index)):
    torque_segment = torque_data[start_index[i]:end_index[i]]
    enspd_segment = enspd_data[start_index[i]:end_index[i]]
    gear_segment = [1 for i in range(0, len(torque_segment))]
    time_segment = [i/fre for i in range(0, len(torque_segment))]
    vehdri_segment = vehdri_data[start_index[i]:end_index[i]]
    vehnon_segment = vehnon_data[start_index[i]:end_index[i]]

    j = 0
    while j < len(torque_segment):
        if torque_segment[j] < 0:
            gear_segment = [gear_segment[k] if k<j else gear_segment[k] + 1 for k in range(0, len(gear_segment))]
            j += fre*2
        else:
            j += 1
    # time zone
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax.plot(time_segment, torque_segment, c='r')
    ax.plot(time_segment, vehdri_segment, c='k')
    ax.plot(time_segment, vehnon_segment, c='k')
    ax2.plot(time_segment, enspd_segment, c='g')
    ax3.plot(time_segment, gear_segment, c='b')
    ax3.set_yticks([])
    ax.set_ylim([-50, 300])
    ax.set_xlim([0, 10])
    ax2.set_ylim([2000, 7000])
    fig.savefig(root_dir + str(i) + '_time')

    # full load curve vs operation points
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 300])
    ax.plot(full_load_curve['speed'].tolist(), full_load_curve['torque'].tolist(), linewidth='5', c='k')
    ax.scatter(enspd_segment, torque_segment, s=10, c=[color_dict[gear_segment[i]] for i in range(0, len(gear_segment))])
    fig.savefig(root_dir + str(i) + '_load')

plt.show()

