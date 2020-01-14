import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = r'C:\Users\lvhui\Desktop\response'
file_list = os.listdir(root_dir)
file_list = [file_list[i] for i in range(0, len(file_list)) if '.csv' in file_list[i]]
dbc = {'torque': 'EnToqActuExtdRngHSC1', 'enspd': 'EnSpdHSC1', 'pedal': 'AccelActuPosHSC1'}
cut_time = 7
fre_dict = {'IP31MCE_65%': 20, 'IP32MCE_50%_不全': 20, 'else': 50}
result_dict = {}

for file_name in file_list:
    data = pd.read_csv(os.path.join(root_dir, file_name), encoding='GB18030')
    torque_data = data[dbc['torque']].tolist()
    enspd_data = data[dbc['enspd']].tolist()
    pedal_data = data[dbc['pedal']].tolist()

    file_name = file_name[0:-4]
    result_dict[file_name] = {}

    try:
        fre = fre_dict[file_name]
    except KeyError:
        fre = fre_dict['else']

    rising_edge = [i for i in range(0, len(pedal_data) - 1) if (pedal_data[i] == 0) & (pedal_data[i+1] != 0)]
    
    for index in rising_edge:
        pedal_ave = np.average(pedal_data[index: index+cut_time*fre + 1])
        enspd_ave = np.average(enspd_data[index: index+cut_time*fre + 1])
        enspd_ave = int(int(enspd_ave/100) * 100)
        if abs(enspd_ave - 1500) <= 100:
            enspd_ave = 1500
        if abs(enspd_ave - 1200) <= 100:
            enspd_ave = 1200
        if enspd_ave > 2000:
            enspd_ave = int(round(enspd_ave/500) * 500)
        if abs(pedal_ave-100) < 3:
            if str(enspd_ave) not in result_dict[file_name]:
                result_dict[file_name][str(enspd_ave)] = {'pedal':[], 'torque':[]}
            result_dict[file_name][str(enspd_ave)]['torque'].append(torque_data[index: index+cut_time*fre + 1])
            result_dict[file_name][str(enspd_ave)]['pedal'].append(pedal_data[index: index+cut_time*fre + 1])


figure_list = []
color_pool = ['black', 'r', 'g', 'y', 'blue']
index=0

# for car in result_dict:
#     try:
#         fre = fre_dict[car]
#     except KeyError:
#         fre = fre_dict['else']
#     print(car)
#     color = color_pool[index]
#     index += 1
#     legend_list.append(car)
#     for enspd in ['1500', '2000', '2500']:
#         if enspd not in figure_list:
#             figure_list.append(enspd)
#             exec('fig_' + enspd + '=plt.figure()')
#             exec('ax_' + enspd + '=fig_' + enspd + '.add_subplot(1,1,1)')
#             exec('ax1_' + enspd + '=ax_' + enspd + '.twinx()')
#             ax_handle = eval('ax_' + enspd)
#             ax1_handle = eval('ax1_' + enspd)
#
#             ax_handle.set_title(enspd + ' rpm')
#             ax_handle.set_xlabel('Time/s')
#             ax_handle.set_ylabel('Torque/Nm')
#             ax_handle.set_xlim(0, cut_time*fre)
#             ax_handle.set_xticklabels([t for t in range(0, cut_time + 1)])
#             ax_handle.set_ylim(-50, 300)
#             ax_handle.set_ylabel('Pedal/%')
#         for i in range(0, len(result_dict[car][enspd]['torque'])):
#             exec('line = ax_' + enspd + '.plot([t for t in range(0, fre*cut_time + 1)], result_dict[car][enspd][\'torque\'][i], color=color)')
#             exec('line = ax_' + enspd + '.plot([t for t in range(0, fre*cut_time + 1)], result_dict[car][enspd][\'pedal\'][i], \':\', color=color)')
# plt.show()

# for direct time vs torque input
#TBD

for enspd in ['1200', '1500', '2000', '2500']:
    index = 0
    exec('fig_' + enspd + '=plt.figure()')
    exec('ax_' + enspd + '=fig_' + enspd + '.add_subplot(1,1,1)')
    exec('ax1_' + enspd + '=ax_' + enspd + '.twinx()')
    ax_handle = eval('ax_' + enspd)
    ax1_handle = eval('ax1_' + enspd)

    ax_handle.set_title(enspd + 'rpm')
    ax_handle.set_xlabel('Time/s')
    ax_handle.set_ylabel('Torque/Nm')
    ax_handle.set_xlim(0, cut_time)
    ax_handle.set_xticklabels([t for t in range(0, cut_time + 1)])
    ax_handle.set_ylim(-50, 300)
    ax1_handle.set_ylim(0, 200)
    ax1_handle.set_ylabel('Pedal/%')
    legend_list = []
    line_list = []

    for car in result_dict:
        try:
            fre = fre_dict[car]
        except KeyError:
            fre = fre_dict['else']
        print(car)
        color = color_pool[index]
        index += 1

        try:
            for i in range(0, len(result_dict[car][enspd]['torque'])):  # all line
            # for i in range(0, 1):
                time = [t/fre for t in range(0, fre*cut_time + 1)]
                exec('line = ax_' + enspd + '.plot(time, result_dict[car][enspd][\'torque\'][i], color=color)')
                exec('ax1_' + enspd + '.plot(time, result_dict[car][enspd][\'pedal\'][i], color=color)')
                if car not in legend_list:
                    line_list += line
                    legend_list.append(car)
        except KeyError:
            pass
    exec('ax_' + enspd + '.legend(line_list, legend_list, loc=\'upper right\', fontsize=10)')
    # ax1.legend(line_list_sort, legend_list_sort, loc='upper right',   fontsize=10)

    # for car in addition_result_dict:
plt.show()