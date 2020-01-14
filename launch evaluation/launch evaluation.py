import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

def smooth(raw_data, window):
    # a: 1-D array containing the data to be smoothed by sliding method
    # WSZ: smoothing window size
    out0 = np.convolve(raw_data, np.ones(window, dtype=int), 'valid') / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(raw_data[:window - 1])[::2] / r
    stop = (np.cumsum(raw_data[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


root_dir = r'C:\Users\lvhui\Desktop\data'
file_list = os.listdir(root_dir)
file_list = [file_name for file_name in file_list if ('.csv' in file_name) and ('result' not in file_name)]

dbc = {'IM31': {'ped': 'AccelActuPosHSC1', 'acc': 'MSLongAccelGHSC', 'gear': 'TrEstdGearHSC1', 'vehspd': 'VehSpdAvgDrvn_H1HSC1'},
       'GL8': {'ped': 'AccelActuPosHSC1', 'acc': 'MSLongAccelGHSC', 'gear': 'TrEstdGearHSC1', 'vehspd': 'GL8_drivespd'},
       'BMW': {'ped': 'BMW_AccPedPos', 'acc': 'LongAccGADJ', 'gear': 'BMW GEAR', 'vehspd': 'BMW_VEHSPD'},
       'IP32MCE': {'ped': 'AccelActuPos_h1HSC1', 'acc': 'LongAccelGHSC', 'gear': 'TrEstdGear_TCMHSC1', 'vehspd': 'VehSpdAvgDrvn_h2HSC1'}}
result_dict = {}
arm_dict = {}
# threshold
fre = 20  # hz
length_min = 2  # s
bump_step = 0.1  # s
# arm_list = [10, 20, 30, 40, 50]
arm_list = [i*5 for i in range(1, 21)]


for file_name in file_list:
    data = pd.read_csv(os.path.join(root_dir, file_name), encoding='GB18030')

    file_name = file_name[:-4]
    arm_dict[file_name] = {}
    result_dict[file_name] = {'delay': [], 'peak_acc': [], 'bump': [], 'peak_time': [], 'ped_ave': []}

    ped_data = data[dbc[file_name]['ped']].tolist()
    acc_data = data[dbc[file_name]['acc']].tolist()
    acc_data = smooth(acc_data, 5)
    gear_data = data[dbc[file_name]['gear']].tolist()
    vehspd_data = data[dbc[file_name]['vehspd']].tolist()
    ped_data = [ped_data[i] if gear_data[i] < 13 else 0 for i in range(0, len(ped_data))]
    ped_data[-1] = 0
    print(file_name + 'acc_offset' + str(np.mean(acc_data)))
    acc_data = [acc_data[i] - np.mean(acc_data) for i in range(0, len(acc_data))]

    rising_edge = [i for i in range(0, len(ped_data)-1) if (ped_data[i] == 0) and (ped_data[i+1] > 0)]
    triling_edge = [i for i in range(1, len(ped_data)) if (ped_data[i-1] > 0) and (ped_data[i] == 0)]

    # validated flag
    length_flag = [1 if (triling_edge[i] - rising_edge[i]) > fre*length_min else 0 for i in range(0, len(rising_edge))]
    rising_edge = [rising_edge[i] for i in range(0, len(length_flag)) if length_flag[i] == 1]
    triling_edge = [triling_edge[i] for i in range(0, len(length_flag)) if length_flag[i] == 1]

    max_min_flag = [1 if (max(ped_data[rising_edge[i]: triling_edge[i]]) - min(ped_data[(rising_edge[i]+int(fre/2)): (triling_edge[i]-int(fre/2))])) < 1 else 0 for i in range(0, len(rising_edge))]
    vehspd_flag = [1 if vehspd_data[rising_edge[i]] < 1 else 0 for i in range(0, len(rising_edge))]
    flag = [vehspd_flag[i]*max_min_flag[i] for i in range(0, len(max_min_flag))]

    rising_edge = [rising_edge[i] for i in range(0, len(flag)) if flag[i] == 1]
    triling_edge = [triling_edge[i] for i in range(0, len(flag)) if flag[i] == 1]

    for i in range(0, len(rising_edge)):
        ped_segement = ped_data[rising_edge[i]: triling_edge[i]]
        acc_segement = acc_data[rising_edge[i]: triling_edge[i]]
        time_segement = [i/fre for i in range(0, len(ped_segement))]
        vehspd_segement = vehspd_data[rising_edge[i]: triling_edge[i]]

        # pedal average
        ped_ave = int(np.average(ped_segement))
        if round(ped_ave/5, 0)*5 in arm_list:
            arm_dict[file_name][int(round(ped_ave/5,0)*5)] = np.array([acc_segement, vehspd_segement])

        # delay
        delay = -1
        delay_acc = 0
        for j in range(0, len(acc_segement)):
            if acc_segement[j] > 0.05:
                delay = time_segement[j]
                delay_acc = round(acc_segement[j], 3)
                break

        peak_acc = max(acc_segement)
        peak_time = time_segement[acc_segement.index(peak_acc)]
        peak_acc = round(peak_acc, 3)

        # bump
        bump_max = 0
        for j in range(0, len(acc_segement)-int(bump_step*fre)):
            if (acc_segement[j+int(bump_step*fre)] - acc_segement[j]) > bump_max:
                bump_max = (acc_segement[j+int(bump_step*fre)] - acc_segement[j])
                bump_max_time = time_segement[j]
        bump = round(bump_max/bump_step, 2)
        print(bump)
        # bump ends

        # store
        result_dict[file_name]['ped_ave'].append(ped_ave)
        result_dict[file_name]['peak_acc'].append(peak_acc)
        result_dict[file_name]['peak_time'].append(peak_time)
        result_dict[file_name]['bump'].append(bump)
        result_dict[file_name]['delay'].append(delay)

        # visulization
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        line1 = ax1.plot(time_segement, acc_segement, c='r', label='acc')
        line2 = ax2.plot(time_segement, ped_segement, c='b', label='ped')
        line3 = ax2.plot(time_segement, vehspd_segement, c='y', label='vehspd')
        line4 = ax1.scatter(delay, delay_acc, c='g')
        line5 = ax1.scatter(peak_time, peak_acc, c='k', marker='x')
        line6 = ax2.plot([bump_max_time, bump_max_time], [0, 100], c='k')
        line7 = ax2.plot([bump_max_time+bump_step, bump_max_time+bump_step], [0, 100], c='k')
        line_list = line1 + line2 + line3
        legend_list = [l.get_label() for l in line_list]
        ax1.legend(line_list, legend_list)
        ax1.set_ylim(-0.1, 0.6)
        ax2.set_ylim(0, 100)
        ax1.set_xlim(0, 4)
        ax1.set_title(str(bump))
        ax1.set_xlabel('time/s')
        ax1.set_ylabel('acc/g')
        ax2.set_ylabel('pedal/%')
        # plt.show()
        fig.savefig(root_dir + '\\' + file_name + '_' + str(i) + '_' + str(ped_ave) + '.jpg')

    output_pd = pd.DataFrame(result_dict[file_name])
    output_pd.to_csv(os.path.join(root_dir, 'result_' + file_name + '.csv'))

workbook = xlsxwriter.Workbook(r"C:\Users\lvhui\Desktop\data\output.xlsx")
worksheet = workbook.add_worksheet('arm')
process_assembly_name = [['velocity']]
# arm plot
legend_list = []
i = 1
for car in arm_dict:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pedal_list = list(arm_dict[car].keys())
    pedal_list.sort()
    for ped in pedal_list:
        if ped in [10, 20, 30, 40, 50, 70, 100]:
            ax.plot(arm_dict[car][ped][1], arm_dict[car][ped][0])
            legend_list.append(ped)
            worksheet.write_row(0, 2*(i-1), ['velocity', ped])
            worksheet.write_column(1, 2*(i-1), arm_dict[car][ped][1])
            worksheet.write_column(1, 2*i-1, arm_dict[car][ped][0])
            i += 1
    ax.set_title(car)
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel('vehspd/kph')
    ax.set_ylabel('acc/g')
    ax.legend(legend_list)
    fig.savefig(root_dir + '\\' + car + '_arm.jpg')

# launch plot
worksheet1 = workbook.add_worksheet('launch')
process_assembly_name = [['time']]
worksheet1.write_row(0, 0, process_assembly_name[0])
worksheet1.write_column(1, 0, [t/fre for t in range(0, 10*fre+1)])
legend_list = []
i = 1
for car in arm_dict:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ped in pedal_list:
        if ped in [10, 20, 30, 40, 50, 70, 100]:
            ax.plot([i/fre for i in range(0, len(arm_dict[car][ped][1]))], arm_dict[car][ped][0])
            legend_list.append(ped)
            worksheet1.write_row(0, i, [ped])
            worksheet1.write_column(1, i, arm_dict[car][ped][0][0: min(fre*10+1, len(arm_dict[car][ped][0]))])
            i += 1
    ax.set_title(car)
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0, 10)
    ax.set_xlabel('time/s')
    ax.set_ylabel('acc/g')
    ax.legend(legend_list)
    fig.savefig(root_dir + '\\' + car + '_launch.jpg')
workbook.close()
