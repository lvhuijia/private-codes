import os
import pickle
import xlsxwriter
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


def phase_offset_correction(signal, fre, offset, right_shift=True):
    if right_shift:
        signal = signal[0:-int(offset*fre)]
        signal = [signal[0]]*int(fre*offset) + signal
    else:
        signal = signal[int(offset*fre):]
        signal = signal + [signal[-1]]*int(fre*offset)
    return signal

# data retrieve
os.chdir(r'C:\Users\吕惠加\Desktop\动力衰减\data and figure')
inputfile = open('result_package.pkl', 'rb')
data_package = pickle.load(inputfile)
inputfile.close()

process_data_assembly = data_package['process_data_assembly']
process_data_title = data_package['process_data_title']
summary_assembly = data_package['summary_assembly']
summary_title = data_package['summary_title']

summary_title_en = ['order', 'vehicle_number', 'temperature_target', 'speed_at_start', 'velocity_at_start', 'velocity_at_end', 'time_for_acc',
                    'pipe_inlet_at_start', 'turbine_outlet_at_start', 'cooling_outlet_at_start', 'cooling_to_valve_at_start', 'valve_inlet_at_start', 'outside_at_start',
                    'pipe_inlet_at_end', 'turbine_outlet_at_end', 'cooling_outlet_at_end', 'cooling_to_valve_at_end', 'valve_inlet_at_end', 'outside_at_end', 'frequency_cal', 'frequency_int']

vehicle_name_trans_dic = {'A18SOV120': 'AP31 CVT', 'A18SOV121': 'AP31 DCT', '卡罗拉': 'COROLLA', '途昂': 'Teramont', '汉兰达': 'HIGHLANDER', 'AS23': 'AS23'}  # for vehicle name mapping

# variable retrieve through title-assemble mapping
for i in range(0, len(summary_title)):
    exec(summary_title_en[i] + '=summary_assembly[i]')

for i in range(0, len(process_data_title)):
    exec(process_data_title[i] + '=process_data_assembly[i]')

# correct the phase shift of specific data: tuang 40C without pre speed and AS23 40C（NO.172）
for order_id in order:
    # if (vehicle_number[order_id] == 'AS23') & (temperature_target[order_id] == '35℃'):
    #     gear[order_id] = phase_offset_correction(gear[order_id], frequency_int[order_id], 1, right_shift=True)
    #     torque[order_id] = phase_offset_correction(torque[order_id], frequency_int[order_id], 1, right_shift=True)
    # if (vehicle_number[order_id] == 'AS23') & (temperature_target[order_id] == '40℃'):
    #     gear[order_id] = phase_offset_correction(gear[order_id], frequency_int[order_id], 1.1, right_shift=False)
    #     torque[order_id] = phase_offset_correction(torque[order_id], frequency_int[order_id], 1.1, right_shift=False)
    #     speed[order_id] = phase_offset_correction(speed[order_id], frequency_int[order_id], 1.1, right_shift=False)

    if order_id in [73,74,75,76,77,78,83,84,85,86,87,88]:
        gear[order_id] = phase_offset_correction(gear[order_id], frequency_int[order_id], 0.8, right_shift=True)
        torque[order_id] = phase_offset_correction(torque[order_id], frequency_int[order_id], 0.8, right_shift=True)
        speed[order_id] = phase_offset_correction(speed[order_id], frequency_int[order_id], 0.8, right_shift=True)
    # velocity[75] = [vel for vel in velocity[77][0:-1]]  # correction fo abnormal velocity


vehicle_number = [vehicle_name_trans_dic[name_origin] for name_origin in vehicle_number]  # vehicle name extract from file name and real name mapping
same_vehicle_flag = False  # for different drawing configuration, if same vehicle in one plot, use line style to distinguish W & W/O pre speed

# wishing_list = [order for order in summary_assembly[0]]
# wishing_list = [[i for i in range(0, 10)], [i for i in range(10, 24)], [i for i in range(24, 36)], [i for i in range(36, 48)],
#                 [i for i in range(48, 61)], [i for i in range(61, 69)], [i for i in range(69, 79)], [i for i in range(79, 89)],
#                 [i for i in range(89, 99)], [i for i in range(99, 111)], [i for i in range(111, 121)], [i for i in range(121, 134)],
#                 [i for i in range(134, 144)], [i for i in range(144, 156)], [i for i in range(156, 166)], [i for i in range(166, 174)],
#                 ]  # 同车同温度

# 有无预转速同图，不同温度分开
# wishing_list = [[3, 6], [17, 21], [28, 32]]  # AP31 CVT 30 35 40
# wishing_list = [[39, 44], [49, 55], [63, 67]]  # AP31 DCT 30 35 40
# wishing_list = [[93, 97], [100, 106], [113, 117]]  # 卡罗拉 30 35 40

# 有无预转速分开，不同温度同图
# wishing_list = [[5, 11, 24], [6, 21, 30]]  # AP31 CVT 30 35 40有无预转速
# wishing_list = [[39, 49, 62], [45, 55, 67]]  # AP31 DCT 30 35 40有无预转速
# wishing_list = [[39, 62], [47, 65]]  # AP31 DCT 30 40有无预转速
# wishing_list = [[93, 100, 113], [97, 106, 117]]  # 卡罗拉 30 35 40有无预转速
# wishing_list = [[146, 158, 169], [151, 162, 170]]  # AP31 DCT 30 40有无预转速
# wishing_list = [[69,70,71,72,79,80,81,82], [73,74,75,76,77,78,83,84,85,86,87,88]]  # AS23 35 40有无预转速 有问题
wishing_list = [[71, 81], [75, 84]]  # AS23 35 40有无预转速 有问题
# wishing_list = [[145, 156, 169], [153, 162, 172]]  # 途昂 30 35 40有无预转速
pre_speed_thre = 1400
wishing_list = [[5, 93], [11, 100], [24, 113]]  # AP31 CVT+卡罗拉 30 35 40有转速
wishing_list = [[6, 97], [21, 106], [30, 117]]  # AP31 CVT+卡罗拉 30 35 40无转速
# wishing_list = [[93, 100, 113], [97, 106, 117]]  #  30 35 40有无预转速

# selected process data output
order_num = len(summary_assembly[0])  # total number of test record
expinfo_num = len(summary_assembly)  # total number of expinfo included in summary
signal_num = len(process_data_assembly)  # total number of signal included in process data
column_num = signal_num + 2 + 1  # including 2 expinfo columns and 1 blank column
sheet_num = len(wishing_list)
workbook = xlsxwriter.Workbook("selected process data.xlsx")
for sheet_id in range(0, sheet_num):
    worksheet = workbook.add_worksheet('sheet ' + str(sheet_id))
    col_id = 0
    for order in wishing_list[sheet_id]:
        worksheet.write_column(0, column_num*col_id, summary_title)  # summary title written in column 0
        worksheet.write_row(0, column_num*col_id+2, process_data_title)  # signal title written in row 0
        row_id = 0
        for expinfo_id in range(0, expinfo_num):  # summary info written in column 1
            worksheet.write_column(row_id, column_num*col_id + 1, [summary_assembly[expinfo_id][order]])
            row_id += 1
        for signal_id in range(0, signal_num):
            worksheet.write_column(1, column_num*col_id + 1 + 1 + signal_id, process_data_assembly[signal_id][order])  # process data written starts from column 2
        col_id += 1
workbook.close()

# figure generation and save
speed_max = 6000
velocity_max = 100
line_color = {'30℃': 'g', '35℃': 'b', '40℃': 'r'}
if same_vehicle_flag:  # data of signle vehicle presented in one figure, use line style to distinguish with/without pre speed
    line_style = {True: '-', False: '--'}
else:
    line_style = {'AP31 CVT': '-', 'AP31 DCT': ':', 'COROLLA': '-.', 'Teramont': ':', 'HIGHLANDER': '-.', 'AS23': '-'}
    line_width = {True: 2, False: 1}   # data of more than one vehicle presented in one figure, use line width to distinguish with/without pre speed

for figure_id in range(0, len(wishing_list)):
    fig1 = plt.figure()  # torque, velocity and speed

    fig1_name = 'fig1 '
    ax11 = fig1.add_subplot(111)
    ax12 = ax11.twinx()

    for order in wishing_list[figure_id]:  # fig name generating, containing the order included
        fig1_name = fig1_name + str(order) + ','
    fig1_name = fig1_name[0:-1]

    torque_max = 0
    gear_max = 0
    power_max = 0
    lines_set = []
    for wish_id in range(0, len(wishing_list[figure_id])):
        order = wishing_list[figure_id][wish_id]
        vehicle_temperautre = vehicle_number[order] + '_' + str(int(outside_at_end[order])) + 'C'
        if same_vehicle_flag:
            # lines for fig1
            line_speed = ax11.plot(time[order], speed[order], linestyle=line_style[speed_at_start[order] > pre_speed_thre], label=vehicle_temperautre,
                                 color=line_color[temperature_target[order]])
            line_torque = ax12.plot(time[order], torque[order], linestyle=line_style[speed_at_start[order] > pre_speed_thre],
                                    color=line_color[temperature_target[order]])
            line_velocity = ax12.plot(time[order], velocity[order], linestyle=line_style[speed_at_start[order] > pre_speed_thre],
                                     color=line_color[temperature_target[order]])

        else:
            # lines for fig1
            line_speed = ax11.plot(time[order], speed[order], linestyle=line_style[vehicle_number[order]], label=vehicle_temperautre,
                                   color=line_color[temperature_target[order]])
            line_torque = ax12.plot(time[order], torque[order], linestyle=line_style[vehicle_number[order]],
                                    color=line_color[temperature_target[order]])
            line_velocity = ax12.plot(time[order], velocity[order], linestyle=line_style[vehicle_number[order]],
                                      color=line_color[temperature_target[order]])

        torque_max = max(torque_max, max(torque[order]))
        lines_set = lines_set + line_speed

    torque_max = math.ceil(torque_max/50)*50
    legend_set = [line.get_label() for line in lines_set]
    ax11.legend(lines_set, legend_set, loc='lower right')
    ax11.grid()
    ax12.grid()

    ax11.set_xlabel('Time(s)')
    ax11.set_ylabel('Engine Speed(rpm)')
    ax11.set_ylim(0, speed_max*3)  # occupy one third of the figure, bottom part
    ax11.set_yticks([i*1000 for i in range(0, int(speed_max/1000)+1)])  # label the speed range with step of 1000
    ax12.set_ylabel('Velocity(kph) & Torque(Nm)')
    ax12.set_ylim(-torque_max/2, torque_max)  # occupy two thirds of the figure, upper part
    ax12.set_yticks([i*50 for i in range(0, int(torque_max/50)+1)])
    plt.savefig(fig1_name+'.png', transparent=True)
    # plt.show()
    # plt.close()

for figure_id in range(0, len(wishing_list)):
    fig2 = plt.figure()  # power, gear(gear ratio) and torque*gear ratio(if available)

    fig2_name = 'fig2 '
    ax_power = HostAxes(fig2, [0.1, 0.08, 0.7, 0.9])  # generate a main axes
    ax_gear = ParasiteAxes(ax_power, sharex=ax_power)  # generate a parasite axes of the main axes, sharing the x axis
    ax_torque_multi_gear = ParasiteAxes(ax_power, sharex=ax_power)
    ax_power.parasites.append(ax_gear)  # claiming the relation of the main and parasite axes
    ax_power.parasites.append(ax_torque_multi_gear)

    ax_power.axis['right'].set_visible(False)
    ax_gear.axis['right'].set_visible(True)
    ax_gear.axis['right'].major_ticklabels.set_visible(True)
    ax_gear.axis['right'].label.set_visible(True)
    if max(gear[wishing_list[figure_id][0]]) > 10:
        torque_mutil_gear_axisline = ax_torque_multi_gear.get_grid_helper().new_fixed_axis  # ax_gear is the twin axes, no need to specify like this
        ax_torque_multi_gear.axis['right2'] = torque_mutil_gear_axisline(loc='right', axes=ax_torque_multi_gear, offset=(40, 0))
    fig2.add_axes(ax_power)

    for order in wishing_list[figure_id]:  # fig name generating, containing the order included
        fig2_name = fig2_name + str(order) + ','
    fig2_name = fig2_name[0:-1]

    gear_max = 0
    power_max = 0
    torque_mutil_gear_max = 0
    lines_set = []
    for wish_id in range(0, len(wishing_list[figure_id])):
        order = wishing_list[figure_id][wish_id]
        vehicle_temperautre = vehicle_number[order] + '_' + str(int(outside_at_end[order])) + 'C'
        if same_vehicle_flag:
            # lines for fig2
            power = [torque[order][i] * speed[order][i]/9550 for i in range(0, len(torque[order]))]
            line_power = ax_power.plot(time[order], power, linestyle=line_style[speed_at_start[order] > pre_speed_thre], label=vehicle_temperautre,
                                 color=line_color[temperature_target[order]])
            line_gear = ax_gear.plot(time[order], gear[order], linestyle=line_style[speed_at_start[order] > pre_speed_thre],
                                     color=line_color[temperature_target[order]])
            if max(gear[order]) > 10:  # if it is gear ratio, plot torque*gear
                torque_multi_gear = [torque[order][i] * gear[order][i] for i in range(0, len(torque[order]))]
                torque_mutil_gear_max = max(torque_mutil_gear_max, max(torque_multi_gear))
                line_torque_multi_gear = ax_torque_multi_gear.plot(time[order], torque_multi_gear, linestyle=line_style[speed_at_start[order] > pre_speed_thre],
                                     color=line_color[temperature_target[order]])
        else:
            # lines for fig2
            power = [torque[order][i] * speed[order][i] / 9550 for i in range(0, len(torque[order]))]
            line_power = ax_power.plot(time[order], power, linestyle=line_style[vehicle_number[order]], label=vehicle_temperautre,
                                   color=line_color[temperature_target[order]])
            line_gear = ax_gear.plot(time[order], gear[order], linestyle=line_style[vehicle_number[order]],
                                  color=line_color[temperature_target[order]])
            if max(gear[order]) > 10:  # if it is gear ratio, plot torque*gear
                torque_multi_gear = [torque[order][i] * gear[order][i] for i in range(0, len(torque[order]))]
                torque_mutil_gear_max = max(torque_mutil_gear_max, max(torque_multi_gear))
                line_torque_multi_gear = ax_torque_multi_gear.plot(time[order], torque_multi_gear, linestyle=line_style[vehicle_number[order]],
                                                                   color=line_color[temperature_target[order]])

        gear_max = max(gear_max, max(gear[order]))
        power_max = max(power_max, max(power))
        lines_set = lines_set + line_power

    legend_set = [line.get_label() for line in lines_set]
    ax_power.legend(lines_set, legend_set, loc='upper left')
    ax_power.set_xlabel('Time(s)')
    ax_power.set_ylabel('Power(kw)')
    ax_power.set_ylim(-20, 2*power_max)
    ax_power.set_yticks([i*20 for i in range(0, int(power_max/20+1))])
    ax_gear.set_ylabel('Gear/Gear ratio')
    ax_gear.set_ylim(-gear_max, gear_max+1)
    ax_gear.set_yticks([i for i in range(0, int(gear_max)+1)])
    if gear_max > 10:  # if it is gear ratio, plot torque*gear
        ax_torque_multi_gear.set_ylabel('Torque*gear ratio(Nm)')
        ax_torque_multi_gear.set_ylim(-torque_mutil_gear_max-50, torque_mutil_gear_max)
        ax_torque_multi_gear.set_yticks([i*500 for i in range(0, int(torque_mutil_gear_max/500)+2)])
        ax_gear.set_ylim(0, 2 * gear_max)
        ax_power.legend(lines_set, legend_set, loc='upper right')
    plt.savefig(fig2_name+'.png', transparent=True)
    # plt.show()


