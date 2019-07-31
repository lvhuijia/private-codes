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
import pandas as pd
from mongo_etl.raw_data_to_mongo import SparseCsvToMgDbBaseClass
from mongo_etl.mongo_data_to_pgsql import MgDbToPgSqlBaseClass
from mongo_etl import update_postgre
from datetime import datetime, timedelta
from mongo_etl.utils.common import lockintime, smooth_wsz, acc_fix, rms_cal
from visulization.models import FiguresData, FeaturesData
from static.py.backend import dbc_car
from mongo_etl import update_mongo
from saic_project_django import settings


class GkTipInOutRetrieve(object):
    def __init__(self, dbc, file, fre, curve_element):
        self.dbc = dbc
        self.file = file
        # self.target_p = target_p
        # self.target_v = target_v
        self.fre = fre
        self.curve_element = curve_element

    def cut_data(self, test_mode=False):
        '''
        Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.
        '''
        data_assembly = pd.read_csv(self.file)

        # parameters setting:
        # for data cutting & filtering
        pedal_constantspeed_min = 0.0
        holdtime_min = 5  # minimum holding time of pedal
        acc_respond_min = 0.01  # minimum acc response in 3s
        pedal_rise_min = 1
        pedal_rise_time_max = 0.3
        pedal_hold_deviation_max = 1.5
        velocity_max_still = 0.0  # maximum velocity when stay still, used for acc offset correctio
        # parameters setting ends

        # info retrieve
        fre = self.fre

        pedal_data = data_assembly[self.dbc['pedal']].tolist()
        velocity_data = data_assembly[self.dbc['velocity']].tolist()
        gear_data = data_assembly[self.dbc['gear']].tolist()
        DR_LD_data = data_assembly[self.dbc['DR_LD']].tolist()
        # SR_X_data = data_assembly[self.dbc['SR_X']].tolist()
        # SR_Y_data = data_assembly[self.dbc['SR_Y']].tolist()
        # SR_Z_data = data_assembly[self.dbc['SR_Z']].tolist()
        # SR_sum_data = [(SR_X_data[i]*SR_X_data[i] + SR_X_data[i]*SR_X_data[i] + SR_X_data[i]*SR_X_data[i])**0.5 for i in range(0, len(pedal_data))]
        acc_data = data_assembly[self.dbc['acc']].tolist()
        speed_data = data_assembly[self.dbc['speed']].tolist()
        abstime = [i*1/fre for i in range(0, len(pedal_data))]
        SR_OA_data = data_assembly[self.dbc['SR_OA']].tolist()

        # # acc signal pre-processing
        # # offset correct, error value removed undo
        # still_flag = (np.array(velocity_data) <= velocity_max_still) & (np.array(pedal_data) == 0) & (np.array(acc_data) < 1)
        # if sum(still_flag) > 20:
        #     acc_offset = sum(np.array(acc_data) * still_flag) / sum(still_flag)
        # else:
        #     acc_offset = np.mean(acc_data)
        #     print('not enough length of data for acc correction: ' + 'only ' + str(sum(still_flag)) + 'points found. Average of all acc are used')
        # acc_correct_CAN = [x - acc_offset for x in acc_data_CAN]

        pedal_var = [pedal_data[i + 1] - pedal_data[i] for i in range(0, len(pedal_data) - 1)]  # pedal variation at each time step
        pedal_var.insert(0, 0)  # 0 at the beginning
        pedal_rising_index = [i - 1 for i in range(0, len(pedal_var)) if pedal_var[i] > 1]
        pedal_rising = [pedal_rising_index[i + 1] for i in range(0, len(pedal_rising_index) - 1) if (
                    pedal_rising_index[i + 1] - pedal_rising_index[i]) > 0.5 * fre]  # index of pedal variation at next time step larger than 1%
        pedal_rising.insert(0, pedal_rising_index[0])
        pedal_rising.append(pedal_rising_index[-1])

        pedal_rising_left = pedal_rising[0:-1]  # start of a test(index)
        pedal_rising_right = [x - 1 for x in pedal_rising[1:]]  # one time step before next start i.e. end of a test(index)
        # after the last test, no new test is started, so the end of the last test is set to be the end of the recording
        pedal_rising_right[-1] = len(pedal_data) - 1

        start = [pedal_rising_left[i] for i in range(0, len(pedal_rising_left)) if
                 (pedal_rising_right[i] - pedal_rising_left[i]) >= holdtime_min * fre]  # time between two rising edges less than 3s is eliminated
        end = [pedal_rising_right[i] for i in range(0, len(pedal_rising_left)) if
               (pedal_rising_right[i] - pedal_rising_left[i]) >= holdtime_min * fre]

        pedal_max = []
        pedal_max_start = []
        pedal_max_end = []
        acc_max_3s = []
        pedal_rise = []
        pedal_min_start = []
        pedal_min_end = []

        # index calculation for data filtering
        for i in range(0, len(start)):
            pedal_cal = pedal_data[start[i]: end[i] + 1]  # test data
            pedal_max.append(max(pedal_cal))
            # fake maximum pedal process(global index)
            pedal_max_process = [x + start[i] for x in range(0, len(pedal_cal)) if abs(pedal_cal[x] - pedal_max[i]) < pedal_hold_deviation_max]
            pedal_max_start.append(pedal_max_process[0])
            pedal_max_end.append(pedal_max_process[-1])

            pedal_min = min(pedal_data[pedal_max_end[i]:end[i] + 1])
            # fake minimum pedal process(global index)
            pedal_min_process = [x for x in range(pedal_max_end[i], end[i] + 1) if abs(pedal_data[x] - pedal_min) < pedal_hold_deviation_max]
            pedal_min_start.append(pedal_min_process[0])
            pedal_min_end.append(pedal_min_process[-1])

            acc_max_3s.append(max(acc_data[start[i]:start[i] + 3 * fre + 1]))
            pedal_rise.append(pedal_max[i] - pedal_cal[0])
        # index calculation for data filtering ends

        # data filtering
        segement_raw = len(start)
        exclude_flag1 = [1 if pedal_data[start[i]] < pedal_constantspeed_min else 0 for i in range(0, segement_raw)]
        exclude_flag2 = [1 if acc_max_3s[i] < acc_respond_min else 0 for i in range(0, segement_raw)]
        exclude_flag3 = [1 if pedal_rise[i] <= pedal_rise_min else 0 for i in range(0, segement_raw)]
        exclude_flag4 = [1 if (pedal_max_start[i] - start[i]) > (pedal_rise_time_max * fre) else 0 for i in range(0, segement_raw)]
        exclude_flag5 = [1 if (pedal_max_end[i] - pedal_max_start[i]) < (holdtime_min * fre) else 0 for i in range(0, segement_raw)]
        exclude_flag6 = [1 if (pedal_max[i] < 35) else 0 for i in range(0, segement_raw)]
        exclude_flag = [max(exclude_flag1[i], exclude_flag2[i], exclude_flag3[i], exclude_flag4[i], exclude_flag5[i], exclude_flag6[i]) for i in
                        range(0, segement_raw)]

        start = [start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        end = [end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_min_start = [pedal_min_start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_min_end = [pedal_min_end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max_start = [pedal_max_start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max_end = [pedal_max_end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max = [pedal_max[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        segement_filtered = len(start)
        # data filtering end

        # tag data and evaluation index calculation.
        parameter_list = []
        for i in range(0, segement_filtered):
            velocity_target = round(velocity_data[start[i]] / 10) * 10
            pedal_target = round(max(pedal_data[start[i]: pedal_min_start[i]]) / 10) * 10
            pedal_target = min(pedal_target, 101)
            # if (velocity_target in [20, 40, 60, 80, 100]) & (pedal_target in [40, 60, 80, 100, 101]):
            #     parameter = str(int(velocity_target)) + 'kph_' + str(int(pedal_target)) + 'ped'
            #     if parameter not in parameter_list:  # select the first segement by default
            #         start_index_can = start[i] - 3*fre
            #         end_index_can = pedal_min_start[i] + 3*fre
            #     parameter_list.append(parameter)

        acc_smooth_vibration = smooth_wsz(acc_data, 10)
        acc_10lowpass_vibration = acc_fix(acc_data, fre, 1e-9, min(10, fre / 2 - 1))
        acc_50lowpass_vibration = acc_fix(acc_data, fre, 1e-9, min(50, fre / 2 - 1))

        # for index calculation and plotting
        acc_delay_min = 0.05
        response_time_thre = 5
        time_plot = 6
        # parameters setting ends

        # data cutting for each segement
        abstime_segement = [abstime[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]
        velocity_segement = [velocity_data[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]
        pedal_segement = [pedal_data[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]
        gear_segement = [gear_data[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]
        speed_segement = [speed_data[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]

        acc_smooth_segement = [acc_smooth_vibration[start[i]:pedal_min_end[i] + 1] for i in
                                         range(0, segement_filtered)]
        acc_10lowpass_segement = [acc_10lowpass_vibration[start[i]:pedal_min_end[i] + 1] for i in
                                            range(0, segement_filtered)]
        acc_50lowpass_segement = [acc_50lowpass_vibration[start[i]:pedal_min_end[i] + 1] for i in
                                            range(0, segement_filtered)]

        SR_segement = [SR_OA_data[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]
        LD_segement = [DR_LD_data[start[i]:pedal_min_end[i] + 1] for i in range(0, segement_filtered)]
        # data cutting ends

        # evaluation index calculation.
        curve_element = self.curve_element
        process_data_dic = {}
        for k in range(0, segement_filtered):
            abstime_cal_CAN = abstime_segement[k]
            velocity_cal = velocity_segement[k]
            pedal_cal = pedal_segement[k]
            speed_cal = speed_segement[k]
            gear_cal = gear_segement[k]
            abstime_cal = abstime_segement[k]
            acc_smooth_cal = acc_smooth_segement[k]
            acc_10lowpass_cal = acc_10lowpass_segement[k]
            acc_50lowpass_cal = acc_50lowpass_segement[k]
            SR_cal = SR_segement[k]
            LD_cal = LD_segement[k]

            velocity_target = round(velocity_cal[0] / 10) * 10
            pedal_target = round(max(pedal_cal) / 10) * 10
            if (velocity_target in [20, 40, 60, 80, 100]) & (pedal_target in [40, 60, 80, 100, 120]):
                name = str(int(velocity_target)) + 'kph ' + str(int(pedal_target)) + '%'
                if name not in curve_element.keys():
                    # index calculation-------------------------------------------------------------------------------
                    # max acc
                    acc_max = max(acc_smooth_cal[0:response_time_thre * fre + 1])

                    # delay
                    delay_time = -1
                    delay_acc = -1  # default value
                    for j in range(0, 3 * fre+1):  # find in 3 seconds
                        if acc_smooth_cal[j] >= acc_delay_min:
                            delay_index = j  # index in vibration group
                            delay_time = round(abstime_cal[delay_index] - abstime_cal[0], 2)
                            delay_acc = acc_smooth_cal[j]
                            break

                    # response time
                    acc_response = 0.95 * acc_max
                    for j in range(0, response_time_thre * fre+1):
                        if acc_smooth_cal[j] >= acc_response:
                            response_index = j  # index in vibration group
                            break
                    response_time = round(abstime_cal[response_index] - abstime_cal[0], 2)
                    response_acc = acc_smooth_cal[j]

                    # 0.1g time
                    zeropointone_time = -1
                    zeropointone_acc = -1
                    for j in range(0, 3 * fre+1):  # find in 3 seconds
                        if acc_smooth_cal[j] >= 0.1:
                            zeropointone_index = j  # index in vibration group
                            zeropointone_time = round(abstime_cal[zeropointone_index] - abstime_cal[0], 2)
                            zeropointone_acc = acc_smooth_cal[j]
                            break

                    # thump rate,start point TBD: maybe maximum acc change point
                    if response_time - delay_time <= 0:
                        thump_rate = -1
                    else:
                        try:
                            thump_rate = round((acc_smooth_cal[response_index] - acc_smooth_cal[delay_index]) / (response_time - delay_time), 2)
                        except UnboundLocalError:
                            thump_rate = round((acc_smooth_cal[response_index] - acc_smooth_cal[0]) / (response_time), 2)

                    # kick calculation:
                    kick_segment = acc_smooth_cal[response_index: response_index + int(0.8 * fre)].tolist()
                    kick_index = [kick_segment.index(max(kick_segment)), kick_segment.index(min(kick_segment))]
                    kick_max = [max(kick_segment), abstime_cal[response_index + kick_index[0]] - abstime_cal[0]]
                    kick_min = [min(kick_segment), abstime_cal[response_index + kick_index[1]] - abstime_cal[0]]
                    if (kick_max[0] - kick_min[0] > 0.05) & (0 < kick_min[1] - kick_max[1] < 0.5) & (kick_max[1] < 3):
                        kick_flag = 'Y_' + str(round(kick_max[0] - kick_min[0], 3))
                    else:
                        kick_flag = 'N'

                    # geardown: input not specific enough

                    # acceleration disturbance
                    acc_disturbance = round(rms_cal(acc_50lowpass_cal[response_index: response_index + 2 * fre]), 3)

                    # vibration dose value

                    # end of index calculation-------------------------------------------------------------------------------

                    # info for all curves integration
                    curve_element[name] = {}
                    curve_element[name]['abstime_vibration'] = abstime_cal[0:time_plot * fre]
                    curve_element[name]['acc'] = acc_smooth_cal[0:time_plot * fre]
                    curve_element[name]['abstime_CAN'] = abstime_cal[0:time_plot * fre]
                    curve_element[name]['pedal'] = pedal_cal[0:time_plot * fre]
                    curve_element[name]['gear'] = gear_cal[0:time_plot * fre]
                    curve_element[name]['speed'] = speed_cal[0:time_plot * fre]
                    curve_element[name]['velocity'] = velocity_cal[0:time_plot * fre]
                    curve_element[name]['SR'] = SR_cal[0:time_plot * fre]
                    curve_element[name]['LD'] = LD_cal[0:time_plot * fre]
                    curve_element[name]['pedal_target'] = pedal_target
                    curve_element[name]['velocity_target'] = velocity_target
                    curve_element[name]['delay'] = [delay_time, delay_acc]
                    curve_element[name]['response'] = [response_time, response_acc]
                    curve_element[name]['zeropointone'] = [zeropointone_time, zeropointone_acc]
                    curve_element[name]['thumprate'] = thump_rate
                    curve_element[name]['kick'] = [kick_flag, kick_max[0], kick_min[0], kick_max[1], kick_min[1]]
                    curve_element[name]['acc disturbance'] = acc_disturbance
                    curve_element[name]['acc max'] = acc_max
                    # info for all curves integration ends

                    # process data integration
                    process_data_dic.update({str(velocity_target) + 'kph ' + str(pedal_target) + '%':
                                                 {'time': curve_element[name]['abstime_vibration'], 'acc': curve_element[name]['acc'],
                                                  'speed': curve_element[name]['speed'], 'velocity': curve_element[name]['velocity'],
                                                  'gear': curve_element[name]['gear'], 'pedal': curve_element[name]['pedal']}
                                             })
                    # process data integration ends

                    # # process data output
                    # workbook = xlsxwriter.Workbook("process data.xlsx")
                    # export_order = ['time', 'acc', 'speed', 'velocity', 'gear', 'pedal']
                    # for keys in process_data_dic:
                    #     worksheet = workbook.add_worksheet(keys)
                    #     process_data = process_data_dic[keys]
                    #     col_id = 0
                    #     for item in export_order:
                    #         worksheet.write_row(0, col_id, [item])
                    #         worksheet.write_column(1, col_id, process_data[item])
                    #         col_id += 1
                    # workbook.close()
        return curve_element

    @staticmethod
    def plot_acc_curve(curve_element, time_plot, gear_max, dump_path):
        gear_flag = 1  # 1:with gear curve,0: without gear curve

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
            exec('fig_' + str(figure_id) + '= plt.figure()')
            ax1 = eval('fig_' + str(figure_id) + '.add_subplot(1, 1, 1)')
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
                figure_index = int(curve_element[key]['velocity_target'] / 20)
                if figure_index == figure_id:
                    # color determination
                    color_index = int(curve_element[key]['pedal_target'] / 20) - 2
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
                        ax1.set_ylim(-0.1, 1.1)
                        ax1.set_yticks([(i-1) * 0.1 for i in range(0, 7)])
                    else:
                        ax2.set_yticks([40 + i * 20 for i in range(0, 4)])
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

            plt.savefig('figure_' + str(figure_id * 20) + '.png', transparent=True)
            plt.show()
            plt.close()

            # index integration
            all_index_dic.update({str(figure_id * 20):
                                      {'pedal': pedal_target_list, 'max acc': acc_max_list, 'delay time': delay_time_list,
                                       'response time': response_time_list, '0.1g time': zeropointone_time_list,
                                       'thump rate': thumprate_list, 'kick': kick_flag_list, 'acc disturbance': acc_disturbance_list}
                                  })

        # index output
        os.chdir(dump_path)
        workbook = xlsxwriter.Workbook("index assembly.xlsx")
        export_order = ['pedal', 'max acc', 'delay time', 'response time', '0.1g time', 'thump rate', 'kick', 'acc disturbance']
        for figure_id in range(1, 6):
            worksheet = workbook.add_worksheet(str(figure_id * 20) + 'kph')
            index_dic = all_index_dic[str(figure_id * 20)]
            index_pd = pd.DataFrame(index_dic)
            index_pd.sort_values(by=['pedal'], inplace=True)
            col_id = 0
            for item in export_order:
                worksheet.write_row(0, col_id, [item])
                worksheet.write_column(1, col_id, index_pd[item])
                col_id += 1
        workbook.close()

    @staticmethod
    def plot_SR_curve(curve_element, time_plot, gear_max, dump_path):
        gear_flag = 1  # 1:with gear curve,0: without gear curve

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
            exec('fig_' + str(figure_id) + '= plt.figure()')
            ax1 = eval('fig_' + str(figure_id) + '.add_subplot(1, 1, 1)')
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
                figure_index = int(curve_element[key]['velocity_target'] / 20)
                if figure_index == figure_id:
                    # color determination
                    color_index = int(curve_element[key]['pedal_target'] / 20) - 2
                    color_index_list.append(color_index)

                    legend_list.append(legend_pool[color_index])
                    curve_element[key]['pedal'] = [100 if curve_element[key]['pedal'][i] == 120 else
                                                   curve_element[key]['pedal'][i] for i in range(0, len(curve_element[key]['pedal']))]

                    abstime_vibration = curve_element[key]['abstime_vibration']
                    reltime_vibration = [abstime_vibration[i] - abstime_vibration[0] for i in range(0, len(abstime_vibration))]
                    abstime_CAN = curve_element[key]['abstime_CAN']
                    reltime_CAN = [abstime_CAN[i] - abstime_CAN[0] for i in range(0, len(abstime_CAN))]

                    pedal_target_list.append(curve_element[key]['pedal_target'])

                    # SR curves
                    line = ax1.plot(reltime_vibration, curve_element[key]['SR'], color=colormap[color_index], linewidth=2)
                    line_list += line
                    ax1.set_xlabel('Time (s)', fontsize=10)
                    ax1.set_ylabel('SR_sum', fontsize=10)
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
                        ax1.set_ylim(-0.2, 4.1)
                        ax1.set_yticks([(i-1) * 0.2 for i in range(0, 12)])
                    else:
                        ax2.set_yticks([40 + i * 20 for i in range(0, 4)])
                        ax2.set_ylabel('Pedal (%)', fontsize=10)
                        ax1.set_ylim(-0.1, 2)
                        ax1.set_yticks([(i-1) * 0.1 for i in range(0, 12)])

            line_legend_df = pd.DataFrame({'line': line_list, 'legend': legend_list, 'color_index': color_index_list})
            line_legend_df_sort = line_legend_df.sort_values(by=['color_index'])
            line_list_sort = line_legend_df_sort['line'].tolist()
            legend_list_sort = line_legend_df_sort['legend'].tolist()

            ax1.legend(line_list_sort, legend_list_sort, loc='upper right', fontsize=10)

            os.chdir(dump_path)
            plt.savefig('SR_figure_' + str(figure_id * 20) + '.png', transparent=True)
            plt.show()
            plt.close()

    @staticmethod
    def plot_LD_curve(curve_element, time_plot, gear_max, dump_path):
        gear_flag = 1  # 1:with gear curve,0: without gear curve

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
            exec('fig_' + str(figure_id) + '= plt.figure()')
            ax1 = eval('fig_' + str(figure_id) + '.add_subplot(1, 1, 1)')
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
                figure_index = int(curve_element[key]['velocity_target'] / 20)
                if figure_index == figure_id:
                    # color determination
                    color_index = int(curve_element[key]['pedal_target'] / 20) - 2
                    color_index_list.append(color_index)

                    legend_list.append(legend_pool[color_index])
                    curve_element[key]['pedal'] = [100 if curve_element[key]['pedal'][i] == 120 else
                                                   curve_element[key]['pedal'][i] for i in range(0, len(curve_element[key]['pedal']))]

                    abstime_vibration = curve_element[key]['abstime_vibration']
                    reltime_vibration = [abstime_vibration[i] - abstime_vibration[0] for i in range(0, len(abstime_vibration))]
                    abstime_CAN = curve_element[key]['abstime_CAN']
                    reltime_CAN = [abstime_CAN[i] - abstime_CAN[0] for i in range(0, len(abstime_CAN))]

                    pedal_target_list.append(curve_element[key]['pedal_target'])

                    # LD curves
                    line = ax1.plot(reltime_vibration, curve_element[key]['LD'], color=colormap[color_index], linewidth=2)
                    line_list += line
                    ax1.set_xlabel('Time (s)', fontsize=10)
                    ax1.set_ylabel('SR_sum', fontsize=10)
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
                        ax1.set_ylim(0, 120)
                        ax1.set_yticks([i * 10 for i in range(0, 7)])
                    else:
                        ax2.set_yticks([40 + i * 20 for i in range(0, 4)])
                        ax2.set_ylabel('Pedal (%)', fontsize=10)
                        ax1.set_ylim(0, 60)
                        ax1.set_yticks([i * 10 for i in range(0, 7)])

            line_legend_df = pd.DataFrame({'line': line_list, 'legend': legend_list, 'color_index': color_index_list})
            line_legend_df_sort = line_legend_df.sort_values(by=['color_index'])
            line_list_sort = line_legend_df_sort['line'].tolist()
            legend_list_sort = line_legend_df_sort['legend'].tolist()

            ax1.legend(line_list_sort, legend_list_sort, loc='upper right', fontsize=10)

            os.chdir(dump_path)
            plt.savefig('LD_figure_' + str(figure_id * 20) + '.png', transparent=True)
            plt.show()
            plt.close()

    def testpara_modify(self):
        for key in self.data_assembly:
            split_list = [i.split('_') for i in self.data_assembly[key]['testparameter']]
            join_list = ['_'.join([i[1], i[0]]) for i in split_list]
            self.data_assembly[key]['testparameter'] = [(i[0:-1] + 'ped') for i in join_list]


class GkTipInOutToPgSQL(MgDbToPgSqlBaseClass):
    pass


class GkTipInOutCal(object):

    def __init__(self, data_dict, fre_can=20, fre_vibration=200):
        self.data_dict = data_dict
        self.df_can = pd.DataFrame(data_dict['can']) if 'can' in self.data_dict.keys() else []
        self.df_vibration = pd.DataFrame(data_dict['vibration']) if 'vibration' in self.data_dict.keys() else []
        self.fre_can = fre_can
        self.fre_vibration = fre_vibration

    def main(self, data_type='tdms'):
        fre_can = self.fre_can
        fre_vir = self.fre_vibration
        data_assembly = self.data_dict

        # for index calculation and plotting
        acc_delay_min = 0.05
        response_time_thre = 5
        time_plot = 10

        if data_type != 'tdms':
            pre_start_time = 3
            gear_data = data_assembly['can']['GearRaw']
            velocity_data = data_assembly['can']['VehSpd_NonDrvn']
            abstime_vibration = data_assembly['vibration']['Time']
            acc_data_vibration = data_assembly['vibration']['IMU_X']
        else:
            pre_start_time = 5
            velocity_data = data_assembly['can']['VehSpdNonDrvn']
            gear_data = data_assembly['can']['GearPos']
            abstime_vibration = data_assembly['voltage']['VoltageTime']
            acc_data_vibration = data_assembly['voltage']['AccX1']

        abstime_CAN = data_assembly['can']['Time']
        pedal_data = data_assembly['can']['AccPedPos']
        speed_data = data_assembly['can']['EnSpd']
        gear_data = [round(x, 2) for x in gear_data]
        para_data = data_assembly['can']['testparameter']

        # gear_max = int(max(gear_data))
        # if gear_max > 9:
        #     gear_data_forward = [x for x in gear_data if x < 10]
        #     gear_max = int(max(gear_data_forward))

        # wavelet decomposition & de-noising：de-noising not found
        acc_smooth_vibration = smooth_wsz(acc_data_vibration, 10).tolist()
        acc_10lowpass_vibration = acc_fix(acc_data_vibration, fre_vir, 1e-9, min(10, int(fre_vir / 2) - 1))
        acc_50lowpass_vibration = acc_fix(acc_data_vibration, fre_vir, 1e-9, min(50, int(fre_vir / 2) - 1))

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(1, 1, 1)
        # ax1.plot(abstime_vibration[100*200:110*200], acc_smooth_vibration[100*200:110*200])
        # ax1.plot(abstime_vibration[100*200:110*200], acc_10lowpass_vibration[100*200:110*200])
        # plt.show()

        # c = pywt.wavedec(acc_correct, wavelet='db4', mode='symmetric', level=2)
        # pywt.threshold(acc_correct, 2, 'soft', 6)
        # 'sym' is the default mode of matlab. db4 same with matlab, dmey not, dmey fits better

        para_data.append('end')
        start_flag = [1 if para_data[i] != para_data[i-1] else 0 for i in range(1, len(para_data))]  # find the start time using testparameter
        start_flag.insert(0, 1)
        start = np.where([x == 1 for x in start_flag])[0].tolist()
        end = [start[i] + min(fre_can*(time_plot+pre_start_time), start[i+1]-start[i]-1) for i in range(0, len(start)-1)]
        start.__delitem__(-1)
        segement_num = len(set(para_data))-1

        # data cutting for each segement
        abstime_segement_CAN = [abstime_CAN[start[i] + fre_can*pre_start_time:end[i]] for i in range(0, segement_num)]
        reltime_segement_CAN = [[x - abstime_segement_CAN[i][0] for x in abstime_segement_CAN[i]] for i in range(0, len(abstime_segement_CAN))]
        velocity_segement = [velocity_data[start[i] + fre_can*pre_start_time:end[i]] for i in range(0, segement_num)]
        pedal_segement = [pedal_data[start[i] + fre_can*pre_start_time:end[i]] for i in range(0, segement_num)]
        gear_segement = [gear_data[start[i] + fre_can*pre_start_time:end[i]] for i in range(0, segement_num)]
        speed_segement = [speed_data[start[i] + fre_can*pre_start_time:end[i]] for i in range(0, segement_num)]

        if data_type == 'tdms':
            start_vibration = np.where([x == 0 for x in abstime_vibration])[0].tolist()  # find the start time using time signal
            end_vibration = [start_vibration[i] + min(fre_vir * (time_plot + pre_start_time), start_vibration[i+1] - start_vibration[i] - 1) for i in range(0, len(start_vibration) - 1)]
            end_vibration.append(min(start_vibration[-1] + fre_vir * (time_plot + pre_start_time), len(acc_smooth_vibration)))
            # end_vibration = [x for x in start_vibration]  # could induce len of acc and CAN signal not coinistant
            # end_vibration.__delitem__(0)
            # end_vibration.append(len(abstime_vibration))
        else:
            start_vibration = [lockintime(x, abstime_CAN, abstime_vibration) for x in start]
            end_vibration = [lockintime(x, abstime_CAN, abstime_vibration) for x in end]
        acc_smooth_segement_vibration = [acc_smooth_vibration[start_vibration[i] + fre_vir*pre_start_time:end_vibration[i]] for i in
                                         range(0, segement_num)]
        acc_10lowpass_segement_vibration = [acc_10lowpass_vibration[start_vibration[i] + fre_vir*pre_start_time:end_vibration[i]] for i in
                                            range(0, segement_num)]
        acc_50lowpass_segement_vibration = [acc_50lowpass_vibration[start_vibration[i] + fre_vir*pre_start_time:end_vibration[i]] for i in
                                            range(0, segement_num)]
        abstime_segement_vibration = [abstime_vibration[start_vibration[i] + fre_vir*pre_start_time:end_vibration[i]] for i in range(0, segement_num)]
        reltime_segement_vibration = [[x - abstime_segement_vibration[i][0] for x in abstime_segement_vibration[i]] for i in range(0, len(abstime_segement_vibration))]

        # time_max1 = max([max(reltime_segement_CAN[i]) for i in range(0, len(reltime_segement_CAN))])
        # time_max2 = max([max(reltime_segement_vibration[i]) for i in range(0, len(reltime_segement_vibration))])

        # data cutting ends

        # 指标计算+组装数据
        pre_trigger = 1  # include 1s before tip in action
        return_data = {'fea_data': {}, 'fig_data': {}}
        for i in range(0, segement_num):
            reltime_segement_CAN[i] = [x - pre_trigger for x in reltime_segement_CAN[i]]  # time shift
            reltime_segement_vibration[i] = [x - pre_trigger for x in reltime_segement_vibration[i]]
            # index calculation-------------------------------------------------------------------------------
            # max acc
            acc_max = round(max(acc_smooth_segement_vibration[i][lockintime(pre_trigger * fre_can, abstime_CAN, abstime_vibration):
                                                                 lockintime(response_time_thre * fre_can, abstime_CAN, abstime_vibration) + 1]), 3)

            # delay
            time_delay = -1
            acc_delay = -1  # default value
            for j in range(0, lockintime(3 * fre_can, abstime_CAN, abstime_vibration)):  # find in 3 seconds
                j += pre_trigger * fre_vir
                if acc_smooth_segement_vibration[i][j] >= acc_delay_min:
                    delay_index = j  # index in vibration group
                    time_delay = round(reltime_segement_vibration[i][j], 2)
                    acc_delay = acc_smooth_segement_vibration[i][j]
                    break

            # response time
            acc_response = 0.95 * acc_max
            for j in range(0, lockintime(response_time_thre * fre_can, abstime_CAN, abstime_vibration) + 1):
                j += pre_trigger * fre_vir
                if acc_smooth_segement_vibration[i][j] >= acc_response:
                    response_index = j  # index in vibration group
                    time_response = round(reltime_segement_vibration[i][j], 2)
                    acc_response = acc_smooth_segement_vibration[i][j]
                    break

            # 0.1g time
            time_zeropointone = -1
            acc_zeropointone = -1
            for j in range(0, lockintime(3 * fre_can, abstime_CAN, abstime_vibration)):  # find in 3 seconds
                j += pre_trigger * fre_vir
                if acc_smooth_segement_vibration[i][j] >= 0.1:
                    zeropointone_index = j  # index in vibration group
                    time_zeropointone = round(reltime_segement_vibration[i][j], 2)
                    acc_zeropointone = acc_smooth_segement_vibration[i][j]
                    break

            # thump rate,start point TBD: maybe maximum acc change point
            if time_response - time_delay <= 0:
                thump_rate = -1
            else:
                thump_rate = round((acc_smooth_segement_vibration[i][response_index] - acc_smooth_segement_vibration[i][delay_index]) / (time_response - time_delay),
                                   2)

            # kick calculation:
            kick_segment = acc_10lowpass_segement_vibration[i][response_index: response_index + lockintime(int(0.8 * fre_can), abstime_CAN, abstime_vibration)]
            kick_index = [kick_segment.index(max(kick_segment)), kick_segment.index(min(kick_segment))]
            kick_max = [max(kick_segment), abstime_segement_vibration[i][response_index + kick_index[0]]]
            kick_min = [min(kick_segment), abstime_segement_vibration[i][response_index + kick_index[1]]]
            if (kick_max[0] - kick_min[0] > 0.05) & (0 < kick_min[1] - kick_max[1] < 0.5) & (kick_max[1] < 3):
                kick_flag = 1
                kick_peak_acc = kick_max[0]
                kick_valley_acc = kick_min[0]
                kick_peak_time = kick_max[1]
                kick_valley_time = kick_min[1]
                kick_does = kick_peak_acc - kick_valley_acc
            else:
                kick_flag = -1
                kick_peak_acc = -1
                kick_valley_acc = -1
                kick_peak_time = -1
                kick_valley_time = -1
                kick_does = -1

            # geardown: input not specific enough

            # acceleration disturbance
            acc_disturbance = round(
                rms_cal(acc_50lowpass_segement_vibration[i][response_index: response_index + lockintime(2 * fre_can, abstime_CAN, abstime_vibration)]), 3)

            # vibration dose value

            # end of index calculation-------------------------------------------------------------------------------
            feature_assembly = [time_delay, acc_delay, time_response, acc_response, time_zeropointone, acc_zeropointone, thump_rate, kick_flag,
                                kick_peak_acc, kick_valley_acc, kick_peak_time, kick_valley_time, kick_does, acc_disturbance]
            feature_assembly = [round(feature,3) for feature in feature_assembly]
            name_assembly = ['time_delay', 'acc_delay', 'time_response', 'acc_response', 'time_zeropointone', 'acc_zeropointone', 'thump_rate',
                             'kick_flag', 'kick_peak_acc', 'kick_valley_acc', 'kick_peak_time', 'kick_valley_time', 'kick_does', 'acc_disturbance']

            if data_type == 'tdms':
                velocity_target = para_data[start[i]].split('_')[0]
                pedal_target = para_data[start[i]].split('_')[1]
            else:
                velocity_target = para_data[start[i]].split('_')[0]
                pedal_target = para_data[start[i]].split('_')[1]
            fig_name = 'TipInOutFig_' + velocity_target
            if fig_name not in return_data['fig_data']:
                return_data['fig_data'][fig_name] = {}
                return_data['fea_data'][fig_name] = {}
                return_data['fig_data'][fig_name]['x'] = []
                return_data['fig_data'][fig_name]['y'] = []
                return_data['fig_data'][fig_name]['p'] = []
                return_data['fea_data'][fig_name]['name'] = []
                return_data['fea_data'][fig_name]['feature'] = []

            return_data['fig_data'][fig_name]['x'] += reltime_segement_CAN[i]
            return_data['fig_data'][fig_name]['y'] += pedal_segement[i]
            return_data['fig_data'][fig_name]['p'] += [pedal_target + '_pedal']*len(reltime_segement_CAN[i])
            return_data['fig_data'][fig_name]['x'] += reltime_segement_CAN[i]
            return_data['fig_data'][fig_name]['y'] += gear_segement[i]
            return_data['fig_data'][fig_name]['p'] += [pedal_target + '_gear']*len(reltime_segement_CAN[i])

            return_data['fig_data'][fig_name]['x'] += reltime_segement_vibration[i]
            return_data['fig_data'][fig_name]['y'] += acc_smooth_segement_vibration[i]
            return_data['fig_data'][fig_name]['p'] += [pedal_target + '_acc']*len(reltime_segement_vibration[i])

            for j in range(0, len(feature_assembly)):
                return_data['fea_data'][fig_name]['name'].append(pedal_target + '_' + name_assembly[j])
                return_data['fea_data'][fig_name]['feature'].append(feature_assembly[j])
        return return_data


def etl_main_tdms(file_path, car_info):
    work_flow = {'extract': True, 'plot': True}

    file_path = r'D:\个人文档\上汽\车型试验\Lamando_14T_DCT_L17SBV002_Banchmark_65%_20190617_results\Comfort\TipInOut'
    # file_path = r'D:\个人文档\上汽\车型试验\ip31_SGE15T_DCT_123123123_SOP_100%_20190528_results\Comfort\TipInOut'

    dbc = {'pedal': 'AccPedPos', 'gear': 'GearPos', 'acc': 'ACC_X', 'velocity': 'VehSpd_NonDrvn', 'DR_LD': 'RR_LD_vs_time',
           'SR_X': 'SR_X', 'SR_Y': 'SR_Y', 'SR_Z': 'SR_Z', 'speed': 'Enspd'}

    if work_flow['extract']:
        file_list = os.listdir(file_path)
        for file_name in file_list:
            print(file_name)


    # step 1：prepare raw data
    if work_flow['prepare_raw_data']:
        print('==============step 1： prepare raw csv data==============')
        tipin_ins = GkTipInOutToMgDB(file_path=file_path, car_info=car_info, gk='TipInOut')  # gk must match the name of the data folder
        tipin_ins.tdms_to_dict()

    #     os.chdir(r'C:\Users\吕惠加\Desktop')
    #     outputfile = open('resample data.pkl', 'wb')
    #     pickle.dump(tipin_ins, outputfile)
    #     outputfile.close()
    #
    # if ~work_flow['prepare_raw_data']:
    #     os.chdir(r'C:\Users\吕惠加\Desktop')
    #     inputfile = open('resample data.pkl', 'rb')
    #     tipin_ins = pickle.load(inputfile)
    #     inputfile.close()

    # step 2: up load to mongoDB
    if work_flow['upload_to_mongo']:
        print('==============step 2: csv -> mongoDB==============')
        # print(len(tipin_ins.data_assembly['Microphone_data']))
        # del tipin_ins.data_assembly['Microphone_data']  # 时间太久
        # del tipin_ins.data_assembly['Vibration_data']
        tipin_ins.testpara_modify()  # DIY your testparameter by that obtained from the file name
        tipin_ins.upload_data_to_mongo(buffer_size=20000)

    # step 3: pre-update postgreSQL
    if work_flow['update_database']:
        update_postgre.update_car_postgre()
        update_postgre.update_car_experiment_postgre()

    # step 4: mongoDB -> postgreSQL
    if work_flow['upload_to_postgre']:
        tipin_ins = GkTipInOutToPgSQL(**work_ins)  # get all the related data from mongodb
        # tipin_ins.retrieve_data(retrieve_dict={'can': ['testparameter', 'Time', 'AccPedPos', 'VehSpd_NonDrvn', 'GearRaw', 'EnSpd'],  # reassemble the signal needed for calculation from data above
        #                                        'voltage': ['testparameter', 'Time', 'IMU_X'],
        #                                        })
        tipin_ins.retrieve_data(retrieve_dict={'can': ['testparameter', 'Time', 'AccPedPos', 'VehSpdNonDrvn', 'GearPos', 'EnSpd'],  # reassemble the signal needed for calculation from data above
                                               'voltage': ['testparameter', 'VoltageTime', 'AccX1'],
                                               })

        ret_data = tipin_ins.cal_and_rate(func_cal=GkTipInOutCal, **{'fre_can': 100, 'fre_vibration': 200})
        print('cal done')

        tipin_ins.insert_into_figures_data_table(fig_data_dict=ret_data['fig_data'])
        print('figure done')

        tipin_ins.insert_into_features_data_table(fea_data_dict=ret_data['fea_data'])
        print('feature done')

    print('finish')


if __name__ == '__main__':
    work_flow = {'extract': True, 'plot': True}

    file_path = r'D:\个人文档\上汽\车型试验\Lamando_14T_DCT_L17SBV002_Banchmark_65%_20190617_results\Comfort\TipInOut'
    # file_path = r'D:\个人文档\上汽\车型试验\ip31_SGE15T_DCT_123123123_SOP_100%_20190528_results\Comfort\TipInOut'

    dbc = {'pedal': 'AccPedPos', 'gear': 'GearPos', 'acc': 'AccX1', 'velocity': 'VehSpdNonDrvn', 'DR_LD': 'RR_LD_vs_time',
           'SR_X': 'SR_X', 'SR_Y': 'SR_Y', 'SR_Z': 'SR_Z', 'speed': 'EnSpd', 'SR_OA': 'SR_OA'}

    if work_flow['extract']:
        file_list = os.listdir(file_path)
        dump_data_path = os.path.join('E:\\dump data\\', file_path.split('\\')[-3])
        dump_data_path = os.path.join(dump_data_path, 'tipin\\')
        os.makedirs(dump_data_path, exist_ok=True)
        curve_element = {}
        for file_name in file_list:
            if 'csv' in file_name:
                print(file_name)
                # para = file_name.split('_')
                # target_p = para[-2][0:-3]
                # target_v = para[-3][0:-3]
                tipin_ins = GkTipInOutRetrieve(dbc=dbc, file=os.path.join(file_path, file_name), fre=100, curve_element=curve_element)
                curve_element = tipin_ins.cut_data()
        # GkTipInOutRetrieve.plot_acc_curve(curve_element=curve_element, time_plot=6, gear_max=6, dump_path=dump_data_path)
        GkTipInOutRetrieve.plot_SR_curve(curve_element=curve_element, time_plot=6, gear_max=6, dump_path=dump_data_path)
        GkTipInOutRetrieve.plot_LD_curve(curve_element=curve_element, time_plot=6, gear_max=6, dump_path=dump_data_path)

    if work_flow['plot']:
        # ret_data = tipin_ins.cal_and_rate(func_cal=GkTipInOutCal, **{'fre_can': 100, 'fre_vibration': 200})
        print('cal done')

    print('finish')
