import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = r'C:\Users\吕惠加\Desktop\response'
file_list = os.listdir(root_dir)
file_list = [file_list[i] for i in range(0, len(file_list)) if '.csv' in file_list[i]]
dbc = {'torque': 'EnToqActuExtdRngHSC1', 'enspd': 'EnSpdHSC1', 'pedal': 'AccelActuPosHSC1'}
cut_time = 7
fre = 20
result_dict = {}

for file_name in file_list:
    data = pd.read_csv(os.path.join(root_dir, file_name), encoding='GB18030')
    torque_data = data[dbc['torque']].tolist()
    enspd_data = data[dbc['enspd']].tolist()
    pedal_data = data[dbc['pedal']].tolist()

    file_name = file_name[0:-4]
    result_dict[file_name] = {}

    rising_edge = [i for i in range(0, len(pedal_data) - 1) if (pedal_data[i] == 0) & (pedal_data[i+1] != 0)]
    
    for index in rising_edge:
        pedal_ave = np.average(pedal_data[index: index+cut_time*fre + 1])
        enspd_ave = np.average(enspd_data[index: index+cut_time*fre + 1])
        enspd_ave = int(enspd_ave/500) * 500
        if abs(pedal_ave-100) < 3:
            if str(enspd_ave) not in result_dict[file_name]:
                result_dict[file_name][str(enspd_ave)] = {'pedal':[], 'torque':[]}
            result_dict[file_name][str(enspd_ave)]['torque'].append(torque_data[index: index+cut_time*fre + 1])
            result_dict[file_name][str(enspd_ave)]['pedal'].append(pedal_data[index: index+cut_time*fre + 1])


figure_list = []
color_pool = ['r', 'g', 'y']
index=0
legend_list = []
for car in result_dict:
    print(car)
    color = color_pool[index]
    index += 1
    legend_list.append(car)
    for enspd in result_dict[car]:
        if enspd not in figure_list:
            figure_list.append(enspd)
            exec('fig_' + enspd + '=plt.figure()')
            exec('ax_' + enspd + '=fig_' + enspd + '.add_subplot(1,1,1)')
            ax_handle = eval('ax_' + enspd)

            ax_handle.set_title(enspd + ' rpm')
            ax_handle.set_xlabel('Time/s')
            ax_handle.set_ylabel('Torque/Nm')
            ax_handle.set_xlim(0, cut_time*fre)
            ax_handle.set_xticklabels([t for t in range(0, cut_time + 1)])
        for i in range(0, len(result_dict[car][enspd]['torque'])):
            exec('line = ax_' + enspd + '.plot([t for t in range(0, fre*cut_time + 1)], result_dict[car][enspd][\'torque\'][i], color=color)')
            exec('line = ax_' + enspd + '.plot([t for t in range(0, fre*cut_time + 1)], result_dict[car][enspd][\'pedal\'][i], \':\', color=color)')
plt.show()

