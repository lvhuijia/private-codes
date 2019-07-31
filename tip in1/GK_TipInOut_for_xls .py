#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt
import pywt
import pickle
import os
import xlsxwriter
import pandas
# from mongo_etl.csv_to_pd import SparseCsvData
from mongo_etl.utils.common import lockintime, smooth_wsz, acc_fix, rms_cal


class TipIn(object):
    """
       Main class of tip in, contains all the thing needed to be calculated or plotted.

       Contains：
       ******
       fun tipin_main————Main function of calculating tipin.
       ******
       fun acc_response————Prepare acceleration response fig data
       fun launch————Prepare launch fig data
       fun max_acc————Prepare maximum acceleration fig data
       fun pedal_map————Prepare pedal map fig data
       fun shift_map————Prepare shift map fig data
       stc fun cut_sg_data_pedal————Data cutting function
       stc fun arm_interpolate————2-D interpolation method to generate system gain arm fig
       ******
       class AccResponse/Launch/MaxAcc/PedalMap/ShiftMap————Wrapper class of the corresponding fig
       class SystemGainDocker————A class that used to wrap all raw data and results
       ******
       fun plot_max_acc————Plotting method of generating maximum acceleration fig. NEED TO BE REWRITE IN Generate_Figs.py
       fun plot_pedal_map————Plotting method of generating pedal map fig. NEED TO BE REWRITE IN Generate_Figs.py
       fun plot_shift_map————Plotting method of generating shift map fig. NEED TO BE REWRITE IN Generate_Figs.py
       """

    def __init__(self, **kwargs):
        """
        Initial function of system gain.

        :param file_path: path of system gain file in local disc
        """
        self.csvinfo = kwargs['csv_info']

    def tipin_main(self):
        '''
        Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.

        '''
        # parameters setting:
        velocity_max_still = 0.0  # maximum velocity when stay still, used for acc offset correction
        # for data cutting $ filtering
        pedal_constantspeed_min = 1
        holdtime_min = 5  # minimum holding time of pedal
        acc_respond_min = 0.05  # minimum acc response in 3s
        pedal_rise_min = 1
        pedal_rise_time_max = 0.3
        pedal_hold_deviation_max = 1.5
        # for index calculation and plotting
        acc_delay_min = 0.05
        response_time_thre = 5
        time_plot = 8
        # parameters setting ends

        # info retrieve
        if isinstance(self.csvinfo, dict):
            fre_can = self.csvinfo['fre_can']  # sample rate
            fre_vir = self.csvinfo['fre_vir']
            data_assembly = self.csvinfo['data_assembly']
        else:
            fre_can = self.csvinfo.re_sample_rate_can  # sample rate
            fre_vir = self.csvinfo.re_sample_rate_vibration
            data_assembly = self.csvinfo.data_assembly

        pedal_data = data_assembly['CAN_data']['AccPedPos'].tolist()
        speed_data = data_assembly['CAN_data']['EnSpd'].tolist()
        acc_data_vibration = data_assembly['Vibration_data']['IMU_X'].tolist()
        acc_data_CAN = data_assembly['CAN_data']['IMU_X_for_correction'].tolist()
        gear_data = data_assembly['CAN_data']['GearRaw'].tolist()
        gear_data = [int(x) for x in gear_data]
        velocity_data = data_assembly['CAN_data']['VehSpd_NonDrvn'].tolist()
        kickdown_data = data_assembly['CAN_data']['GPS_Heading']
        abstime_CAN = data_assembly['CAN_data']['Time']
        abstime_vibration = data_assembly['Vibration_data']['Time']
        pedal_data = [120 if kickdown_data[i] == 1 else pedal_data[i] for i in range(0, len(pedal_data))]
        gear_max = int(max(gear_data))
        if 'others_1' in sparsecsv['data_assembly']['CAN_data']:
            others_1 = sparsecsv['data_assembly']['CAN_data']['others_1']
        else:
            others_1 = data_assembly['CAN_data']['GPS_Heading']

        if isinstance(self.csvinfo, dict):
            kickdown_data = kickdown_data.tolist()
            abstime_CAN = abstime_CAN.tolist()
            abstime_vibration = abstime_vibration.tolist()
            others_1 = others_1.tolist()

        if gear_max > 9:
            gear_data_forward = [x for x in gear_data if x < 10]
            gear_max = int(max(gear_data_forward))

        # acc signal pre-processing
        # offset correct, error value removed undo
        still_flag = (np.array(velocity_data) <= velocity_max_still) & (np.array(pedal_data) == 0) & (np.array(acc_data_CAN) < 1)
        acc_offset = sum(np.array(acc_data_CAN) * still_flag)/sum(still_flag)
        acc_correct_vibration = [x-acc_offset for x in acc_data_vibration]
        acc_correct_CAN = [x-acc_offset for x in acc_data_CAN]

        # wavelet decomposition & de-noising：de-noising not found
        acc_smooth_vibration = smooth_wsz(acc_correct_vibration, 5)
        acc_10lowpass_vibration = acc_fix(acc_correct_vibration, fre_vir, 1e-9, min(10, fre_vir/2-1))
        acc_50lowpass_vibration = acc_fix(acc_correct_vibration, fre_vir, 1e-9, min(50, fre_vir/2-1))
        if 'others_1' in sparsecsv['data_assembly']['CAN_data']:
            acc_10lowpass_vibration = smooth_wsz(acc_correct_vibration, 5).tolist()
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(1, 1, 1)
        # ax1.plot(abstime_vibration[100*200:110*200], acc_smooth_vibration[100*200:110*200])
        # ax1.plot(abstime_vibration[100*200:110*200], acc_10lowpass_vibration[100*200:110*200])
        # plt.show()

        # c = pywt.wavedec(acc_correct, wavelet='db4', mode='symmetric', level=2)
        # pywt.threshold(acc_correct, 2, 'soft', 6)
        # 'sym' is the default mode of matlab. db4 same with matlab, dmey not, dmey fits better

        pedal_var = [pedal_data[i+1]-pedal_data[i] for i in range(0, len(pedal_data)-1)]  # pedal variation at each time step
        pedal_var.insert(0, 0)  # 0 at the beginning
        pedal_rising_index = [i-1 for i in range(0, len(pedal_var)) if pedal_var[i] > 1]
        pedal_rising = [pedal_rising_index[i+1] for i in range(0, len(pedal_rising_index)-1) if (pedal_rising_index[i+1] - pedal_rising_index[i]) > 0.5*fre_can]  # index of pedal variation at next time step larger than 1%
        pedal_rising.insert(0, pedal_rising_index[0])
        pedal_rising.append(pedal_rising_index[-1])

        pedal_rising_left = pedal_rising[0:-1]  # start of a test(index)
        pedal_rising_right = [x - 1 for x in pedal_rising[1:]]  # one time step before next start i.e. end of a test(index)
        pedal_rising_right[-1] = len(pedal_data)-1  # after the last test, no new test is started, so the end of the last test is set to be the end of the recording

        start = [pedal_rising_left[i] for i in range(0, len(pedal_rising_left)) if (pedal_rising_right[i] - pedal_rising_left[i]) >= holdtime_min*fre_can]  # time between two rising edges less than 3s is eliminated
        end = [pedal_rising_right[i] for i in range(0, len(pedal_rising_left)) if (pedal_rising_right[i] - pedal_rising_left[i]) >= holdtime_min*fre_can]

        pedal_max = []
        pedal_max_start = []
        pedal_max_end = []
        acc_max_3s = []
        pedal_rise = []
        pedal_min_start = []
        pedal_min_end = []

        # index calculation for data filtering
        for i in range(0, len(start)):
            pedal_cal = pedal_data[start[i]: end[i]+1]  # test data
            pedal_max.append(max(pedal_cal))
            # fake maximum pedal process(global index)
            pedal_max_process = [x + start[i] for x in range(0, len(pedal_cal)) if abs(pedal_cal[x] - pedal_max[i]) < pedal_hold_deviation_max]
            pedal_max_start.append(pedal_max_process[0])
            pedal_max_end.append(pedal_max_process[-1])

            pedal_min = min(pedal_data[pedal_max_end[i]:end[i]+1])
            # fake minimum pedal process(global index)
            pedal_min_process = [x for x in range(pedal_max_end[i], end[i]+1) if abs(pedal_data[x] - pedal_min) < pedal_hold_deviation_max]
            pedal_min_start.append(pedal_min_process[0])
            pedal_min_end.append(pedal_min_process[-1])

            acc_max_3s.append(max(acc_correct_CAN[start[i]:start[i] + 3*fre_can+1]))
            pedal_rise.append(pedal_max[i] - pedal_cal[0])
        # index calculation for data filtering ends

        # data filtering
        segement_raw = len(start)
        exclude_flag1 = [1 if pedal_data[start[i]] < pedal_constantspeed_min else 0 for i in range(0, segement_raw)]
        exclude_flag2 = [1 if acc_max_3s[i] < acc_respond_min else 0 for i in range(0, segement_raw)]
        exclude_flag3 = [1 if pedal_rise[i] <= pedal_rise_min else 0 for i in range(0, segement_raw)]
        exclude_flag4 = [1 if (pedal_max_start[i] - start[i]) > (pedal_rise_time_max*fre_can) else 0 for i in range(0, segement_raw)]
        exclude_flag5 = [1 if (pedal_max_end[i] - pedal_max_start[i]) < (holdtime_min*fre_can) else 0 for i in range(0, segement_raw)]
        exclude_flag6 = [1 if (pedal_max[i] < 35) else 0 for i in range(0, segement_raw)]
        exclude_flag = [max(exclude_flag1[i], exclude_flag2[i], exclude_flag3[i], exclude_flag4[i], exclude_flag5[i], exclude_flag6[i]) for i in range(0, segement_raw)]

        start = [start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        end = [end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_min_start = [pedal_min_start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_min_end = [pedal_min_end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max_start = [pedal_max_start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max_end = [pedal_max_end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max = [pedal_max[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        segement_filtered = len(start)
        # data filtering end

        # data cutting for each segement
        abstime_segement_CAN = [abstime_CAN[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        velocity_segement = [velocity_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        pedal_segement = [pedal_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        gear_segement = [gear_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        speed_segement = [speed_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        start_vibration = [lockintime(x, abstime_CAN, abstime_vibration) for x in start]
        pedal_min_end_vibration = [lockintime(x, abstime_CAN, abstime_vibration) for x in pedal_min_end]
        acc_smooth_segement_vibration = [acc_smooth_vibration[start_vibration[i]:pedal_min_end_vibration[i]+1] for i in range(0, segement_filtered)]
        acc_10lowpass_segement_vibration = [acc_10lowpass_vibration[start_vibration[i]:pedal_min_end_vibration[i]+1] for i in range(0, segement_filtered)]
        acc_50lowpass_segement_vibration = [acc_50lowpass_vibration[start_vibration[i]:pedal_min_end_vibration[i]+1] for i in range(0, segement_filtered)]
        abstime_segement_vibration = [abstime_vibration[start_vibration[i]:pedal_min_end_vibration[i]+1] for i in range(0, segement_filtered)]
        others_1_segement = [others_1[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]

        # data cutting ends

        # evaluation index calculation.
        curve_element = {}
        process_data_dic = {}
        for i in range(0, segement_filtered):
            abstime_cal_CAN = abstime_segement_CAN[i]
            velocity_cal = velocity_segement[i]
            pedal_cal = pedal_segement[i]
            speed_cal = speed_segement[i]
            gear_cal = gear_segement[i]
            abstime_cal_vibration = abstime_segement_vibration[i]
            acc_smooth_cal_vibration = acc_smooth_segement_vibration[i]
            acc_10lowpass_cal_vibration = acc_10lowpass_segement_vibration[i]
            acc_50lowpass_cal_vibration = acc_50lowpass_segement_vibration[i]
            others_1_cal = others_1_segement[i]

            velocity_target = round(velocity_cal[0]/10)*10
            pedal_target = round(max(pedal_cal)/10)*10
            if (velocity_target in [20, 40, 60, 80, 100]) & (pedal_target in [40, 60, 80, 100, 120]):
                name = str(int(velocity_target)) + 'kph ' + str(int(pedal_target)) + '%'
                if ~(name in curve_element.keys()) or ((name in curve_element.keys()) and (len(acc_smooth_cal_vibration) > len(curve_element[name]['acc']))):
                    # index calculation-------------------------------------------------------------------------------
                    # max acc
                    acc_max = round(max(acc_smooth_cal_vibration[0:lockintime(response_time_thre*fre_can, abstime_CAN, abstime_vibration)+1]), 3)

                    # delay
                    delay_time = -1
                    delay_acc = -1  # default value
                    for j in range(0, lockintime(3*fre_can, abstime_CAN, abstime_vibration)):  # find in 3 seconds
                        if acc_smooth_cal_vibration[j] >= acc_delay_min:
                            delay_index = j  # index in vibration group
                            delay_time = round(abstime_cal_vibration[delay_index] - abstime_cal_vibration[0], 2)
                            delay_acc = acc_smooth_cal_vibration[j]
                            break

                    # response time
                    acc_response = 0.95*acc_max
                    for j in range(0, lockintime(response_time_thre*fre_can, abstime_CAN, abstime_vibration)+1):
                        if acc_smooth_cal_vibration[j] >= acc_response:
                            response_index = j  # index in vibration group
                            break
                    response_time = round(abstime_cal_vibration[response_index] - abstime_cal_vibration[0], 2)
                    response_acc = acc_smooth_cal_vibration[j]

                    # 0.1g time
                    zeropointone_time = -1
                    zeropointone_acc = -1
                    for j in range(0, lockintime(3*fre_can, abstime_CAN, abstime_vibration)):  # find in 5 seconds
                        if acc_smooth_cal_vibration[j] >= 0.1:
                            zeropointone_index = j  # index in vibration group
                            zeropointone_time = round(abstime_cal_vibration[zeropointone_index] - abstime_cal_vibration[0], 2)
                            zeropointone_acc = acc_smooth_cal_vibration[j]
                            break

                    # thump rate,start point TBD: maybe maximum acc change point
                    if response_time - delay_time <= 0:
                        thump_rate = -1
                    else:
                        thump_rate = round((acc_smooth_cal_vibration[response_index] - acc_smooth_cal_vibration[delay_index])/(response_time - delay_time), 2)

                    # kick calculation:
                    kick_segment = acc_10lowpass_cal_vibration[response_index: response_index+lockintime(int(1.5*fre_can), abstime_CAN, abstime_vibration)]
                    kick_index = [kick_segment.index(max(kick_segment)), kick_segment.index(min(kick_segment))]
                    kick_max = [max(kick_segment), abstime_cal_vibration[response_index+kick_index[0]]-abstime_cal_vibration[0]]
                    kick_min = [min(kick_segment), abstime_cal_vibration[response_index+kick_index[1]]-abstime_cal_vibration[0]]
                    if (kick_max[0] - kick_min[0] > 0.05) & (0 < kick_min[1] - kick_max[1] < 0.5):
                        kick_flag = 'Y_' + str(round(kick_max[0] - kick_min[0], 3))
                    else:
                        kick_flag = 'N'

                    # geardown: input not specific enough

                    # acceleration disturbance
                    acc_disturbance = round(rms_cal(acc_50lowpass_cal_vibration[response_index: response_index + lockintime(2*fre_can, abstime_CAN, abstime_vibration)]), 3)

                    # vibration dose value

                    # end of index calculation-------------------------------------------------------------------------------

                    # info for all curves integration
                    curve_element[name] = {}
                    curve_element[name]['abstime_vibration'] = abstime_cal_vibration[0:lockintime(time_plot*fre_can+1, abstime_CAN, abstime_vibration)]
                    curve_element[name]['acc'] = acc_smooth_cal_vibration[0:lockintime(time_plot*fre_can+1, abstime_CAN, abstime_vibration)]
                    curve_element[name]['abstime_CAN'] = abstime_cal_CAN[0:time_plot*fre_can+1]
                    curve_element[name]['pedal'] = pedal_cal[0:time_plot*fre_can+1]
                    curve_element[name]['gear'] = gear_cal[0:time_plot*fre_can+1]
                    curve_element[name]['speed'] = speed_cal[0:time_plot*fre_can+1]
                    curve_element[name]['velocity'] = velocity_cal[0:time_plot*fre_can+1]
                    curve_element[name]['pedal_target'] = pedal_target
                    curve_element[name]['velocity_target'] = velocity_target
                    curve_element[name]['delay'] = [delay_time, delay_acc]
                    curve_element[name]['response'] = [response_time, response_acc]
                    curve_element[name]['zeropointone'] = [zeropointone_time, zeropointone_acc]
                    curve_element[name]['thumprate'] = thump_rate
                    curve_element[name]['kick'] = [kick_flag, kick_max[0], kick_min[0], kick_max[1], kick_min[1]]
                    curve_element[name]['acc disturbance'] = acc_disturbance
                    curve_element[name]['acc max'] = acc_max

                    curve_element[name]['others_1'] = others_1_cal[0:time_plot*fre_can+1]

                    # info for all curves integration ends

                    # process data integration
                    process_data_dic.update({str(velocity_target) + 'kph ' + str(pedal_target) + '%':
                                            {'time': curve_element[name]['abstime_vibration'], 'acc': curve_element[name]['acc'],
                                             'speed_rear': curve_element[name]['speed'], 'speed_front': curve_element[name]['others_1'],
                                             'velocity':  curve_element[name]['velocity'],
                                             'gear':  curve_element[name]['gear'], 'pedal': curve_element[name]['pedal']}
                                             })
                    # process data integration ends

                    # process data output
                    workbook = xlsxwriter.Workbook("process data.xlsx")
                    export_order = ['time', 'acc', 'speed_rear', 'speed_front', 'velocity', 'gear', 'pedal']
                    for keys in process_data_dic:
                        worksheet = workbook.add_worksheet(keys)
                        process_data = process_data_dic[keys]
                        col_id = 0
                        for item in export_order:
                            worksheet.write_row(0, col_id, [item])
                            worksheet.write_column(1, col_id, process_data[item])
                            col_id += 1
                    workbook.close()

        TipIn.plot_acc_curve(self, curve_element, time_plot, gear_max)

    def plot_acc_curve(self, curve_element, time_plot, gear_max):
        gear_flag = 0  # 1:with gear curve,0: without gear curve

        # colormap
        colormap = []
        colormap.append([0.8431, 0.6781, 0.2941])
        colormap.append([0.9061, 0.5491, 0.2241])
        colormap.append([0.7291, 0.4351, 0.2351])
        colormap.append([0.5181, 0.2631, 0.1531])
        colormap.append([0.3101, 0.3181, 0.1651])
        colormap_for_index = ['r', 'r', 'g', 'g', 'b', 'b', 'k', 'k']
        legend_pool = ['40%', '60%', '80%', '100%', 'kickdown']

        all_index_dic = {}
        for figure_id in range(1, 6):
            exec('self.fig_' + str(figure_id) +'= plt.figure()')
            ax1 = eval('self.fig_' + str(figure_id) + '.add_subplot(1, 1, 1)')
            color_index_list = []
            line_list = []
            legend_list = []
            pedal_target_list = []
            delay_time_list = []
            delay_acc_list = []
            response_time_list = []
            response_acc_list = []
            zeropointone_time_list = []
            zeropointone_acc_list = []
            thumprate_list = []
            kick_flag_list = []
            kick_value_list = []
            kick_time_list = []
            acc_disturbance_list = []
            acc_max_list = []

            for key in curve_element:
                figure_index = int(curve_element[key]['velocity_target']/20)
                if figure_index == figure_id:
                    # color determination
                    color_index = int(curve_element[key]['pedal_target']/20)-2
                    color_index_list.append(color_index)

                    legend_list.append(legend_pool[color_index])
                    curve_element[key]['pedal'] = [100 if curve_element[key]['pedal'][i] == 120 else
                                                   curve_element[key]['pedal'][i] for i in range(0, len(curve_element[key]['pedal']))]

                    abstime_vibration = curve_element[key]['abstime_vibration']
                    reltime_vibration = [abstime_vibration[i] - abstime_vibration[0] for i in range(0, len(abstime_vibration))]
                    abstime_CAN = curve_element[key]['abstime_CAN']
                    reltime_CAN = [abstime_CAN[i] - abstime_CAN[0] for i in range(0, len(abstime_CAN))]

                    pedal_target_list.append(curve_element[key]['pedal_target'])
                    delay_time_list.append(curve_element[key]['delay'][0])
                    delay_acc_list.append(curve_element[key]['delay'][1])
                    response_time_list.append(curve_element[key]['response'][0])
                    response_acc_list.append(curve_element[key]['response'][1])
                    zeropointone_time_list.append(curve_element[key]['zeropointone'][0])
                    zeropointone_acc_list.append(curve_element[key]['zeropointone'][1])
                    thumprate_list.append(curve_element[key]['thumprate'])
                    kick_flag_list.append(curve_element[key]['kick'][0])
                    if curve_element[key]['kick'][0][0] == 'Y':
                        kick_value_list += curve_element[key]['kick'][1: 3]
                        kick_time_list += curve_element[key]['kick'][3: 5]
                    acc_disturbance_list.append(curve_element[key]['acc disturbance'])
                    acc_max_list.append(curve_element[key]['acc max'])

                    # acc curves
                    line = ax1.plot(reltime_vibration, curve_element[key]['acc'], color=colormap[color_index], linewidth=2)
                    line_list += line
                    ax1.set_xlabel('Time (s)', fontsize=10)
                    ax1.set_ylabel('Acc (g)', fontsize=10)
                    ax1.set_title(str(int(curve_element[key]['velocity_target'])) + 'kph', fontsize=12)
                    ax1.set_xlim(0, time_plot)

                    # pedal curves
                    ax2 = ax1.twinx()
                    ax2.plot(reltime_CAN, curve_element[key]['pedal'], color=colormap[color_index])
                    ax2.set_ylim(0, 101)

                    # gear curves if requested, setting the second y axis label accordingly
                    if gear_flag == 1:
                        ax3 = ax1.twinx()
                        ax3.plot(reltime_CAN, curve_element[key]['gear'], ':', color=colormap[color_index], linewidth=3)
                        ax3.set_ylim(-gear_max - 1, gear_max + 1)
                        ax3.set_yticks([i for i in range(0, gear_max + 2)])
                        ax3.set_ylabel('Gear', fontsize=10)
                        ax2.set_yticks([])
                        ax1.set_ylim(0, 1)
                        ax1.set_yticks([i * 0.1 for i in range(0, 6)])
                    else:
                        ax2.set_yticks([40+i*20 for i in range(0, 4)])
                        ax2.set_ylabel('Pedal (%)', fontsize=10)
                        ax1.set_ylim(0, 0.8)
                        ax1.set_yticks([i * 0.1 for i in range(0, 9)])

            line_legend_df = pd.DataFrame({'line': line_list, 'legend': legend_list, 'color_index': color_index_list})
            line_legend_df_sort = line_legend_df.sort_values(by=['color_index'])
            line_list_sort = line_legend_df_sort['line'].tolist()
            legend_list_sort = line_legend_df_sort['legend'].tolist()

            # index visualization
            line = ax1.scatter(delay_time_list, delay_acc_list, c='r', marker='o', s=40)
            line_list_sort.append(line)
            legend_list_sort.append('delay')
            line = ax1.scatter(response_time_list, response_acc_list, c='g', marker='^', s=40)
            line_list_sort.append(line)
            legend_list_sort.append('response')
            line = ax1.scatter(zeropointone_time_list, zeropointone_acc_list, c='b', marker='x', s=40)
            line_list_sort.append(line)
            legend_list_sort.append('0.1g')

            for i in range(0, len(kick_time_list)):
                ax1.plot([kick_time_list[i] - 0.2, kick_time_list[i] + 0.2], [kick_value_list[i], kick_value_list[i]],
                         c=colormap_for_index[i], linewidth=2)

            ax1.legend(line_list_sort, legend_list_sort, loc='upper right', fontsize=10)

            plt.savefig('figure_' + str(figure_id*20) + '.png', transparent=True)
            plt.show()
            plt.close()

            # index integration
            all_index_dic.update({str(figure_id*20):
                                 {'pedal': pedal_target_list, 'max acc': acc_max_list, 'delay time': delay_time_list, 'response time': response_time_list, '0.1g time': zeropointone_time_list,
                                 'thump rate': thumprate_list, 'kick': kick_flag_list, 'acc disturbance': acc_disturbance_list}
                                  })

        # index output
        workbook = xlsxwriter.Workbook("index assembly.xlsx")
        export_order = ['pedal', 'max acc', 'delay time', 'response time', '0.1g time', 'thump rate', 'kick', 'acc disturbance']
        for figure_id in range(1, 6):
            worksheet = workbook.add_worksheet(str(figure_id*20) + 'kph')
            index_dic = all_index_dic[str(figure_id*20)]
            index_pd = pd.DataFrame(index_dic)
            index_pd.sort_values(by=['pedal'], inplace=True)
            col_id = 0
            for item in export_order:
                worksheet.write_row(0, col_id, [item])
                worksheet.write_column(1, col_id, index_pd[item])
                col_id += 1
        workbook.close()


if __name__ == '__main__':
    os.chdir(r'C:\Users\吕惠加\Desktop')
    # inputfile = open('resample data.pkl', 'rb')
    # sparsecsv = pickle.load(inputfile)
    # inputfile.close()

    sparsecsv = {'data_assembly': {'CAN_data': {}, 'Vibration_data': {}}, 'fre_can': 20, 'fre_vir': 20}
    tipin_Data_ful = pd.read_csv('D:/个人文档/上汽/车型试验/Tesla/20180906_200232(350_8)tipinout_标准模式_1.csv',encoding='gbk')
    tipin_Data_Selc = tipin_Data_ful.loc[:, ['Tesla_AccPedPos', 'Tesla_rear_MotorSpeed_LF', 'Tesla_Frontmotorspeed_LF', 'MSLongAccelGHSC',
                                             'Tesla_GearPos', 'Tesla_VehSpd_LF', '方向', 'time']]
    sparsecsv['data_assembly']['CAN_data']['AccPedPos'] = tipin_Data_Selc['Tesla_AccPedPos']
    sparsecsv['data_assembly']['CAN_data']['EnSpd'] = tipin_Data_Selc['Tesla_rear_MotorSpeed_LF']
    sparsecsv['data_assembly']['Vibration_data']['IMU_X'] = tipin_Data_Selc['MSLongAccelGHSC']
    sparsecsv['data_assembly']['CAN_data']['IMU_X_for_correction'] = tipin_Data_Selc['MSLongAccelGHSC']
    sparsecsv['data_assembly']['CAN_data']['GearRaw'] = tipin_Data_Selc['Tesla_GearPos']
    sparsecsv['data_assembly']['CAN_data']['VehSpd_NonDrvn'] = tipin_Data_Selc['Tesla_VehSpd_LF']
    sparsecsv['data_assembly']['CAN_data']['GPS_Heading'] = tipin_Data_Selc['方向']
    sparsecsv['data_assembly']['CAN_data']['Time'] = tipin_Data_Selc['time']
    sparsecsv['data_assembly']['Vibration_data']['Time'] = tipin_Data_Selc['time']
    sparsecsv['data_assembly']['CAN_data']['others_1'] = tipin_Data_Selc['Tesla_Frontmotorspeed_LF']

    a = TipIn(csv_info=sparsecsv)
    a.tipin_main()
    print('Finish!')
