#!/usr/bin/env python
# -*- coding:utf-8 -*-

import mdfreader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xlwt
import xlsxwriter
import xlrd
from itertools import groupby


def clear_file_folder(file_path):
    file_list = os.listdir(file_path)
    for file in file_list:
        os.remove(os.path.join(file_path, file))
    print(file_path + ' folder was clear')


def draw_operation_map(mask, cycle_operation, xx, yy, fig_title, fig_name, save_folder):
    operation_map = np.zeros(xx.shape)
    for i in range(0, len(mask)):
        if mask[i]:
            operation_map += cycle_operation[i]
    operation_map = operation_map / sum(mask)

    fig = plt.pcolor(xx, yy, operation_map, cmap='Reds')  # Reds
    fig.set_clim(vmin=0, vmax=120)
    plt.colorbar(fig)
    plt.title(fig_title)
    plt.grid(True)
    os.chdir(save_folder)
    plt.savefig(fig_name, transparent=True)
    plt.close()
    return operation_map


def draw_hybrid_mode_pie(mask, cycle_hybrid_mode, label_list, color_list, fig_title, save_folder, fig_name):
    hybrid_mode_ditribution = [0] * 9
    for i in range(0, len(mask)):
        if mask[i]:
            for j in range(0, 9):
                hybrid_mode_ditribution = [hybrid_mode_ditribution[k] + cycle_hybrid_mode[i][k] for k in range(0, 9)]
    label = [label_list[i] for i in range(0, len(label_list)) if hybrid_mode_ditribution[i] > 0]
    color = [color_list[i] for i in range(0, len(color_list)) if hybrid_mode_ditribution[i] > 0]
    size = [x for x in hybrid_mode_ditribution if x > 0]
    explode = [0.1] * len(label)
    plt.pie(size, labels=label, startangle=50, autopct='%1.1f%%', explode=explode, colors=color)
    plt.title(fig_title)
    os.chdir(save_folder)
    plt.savefig(fig_name)
    plt.close()
    return hybrid_mode_ditribution


class FuelConsumption(object):

    def __init__(self, **kwargs):
        self.filepath = kwargs['root_path']
        self.range = kwargs['work_range']
        self.dbc = kwargs['dbc_dict']

    def fc_main(self):
        # settings--------------------------------------------
        fre = 20
        thre_value = 0.5
        mesh_grid_enspd = 200
        mesh_grid_toq = 10
        x = np.arange(0, 5000, mesh_grid_enspd)
        y = np.arange(0, 260, mesh_grid_toq)
        xx, yy = np.meshgrid(x, y)
        dbc = self.dbc
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        vel_step = 10
        vel_start = 0
        vel_stop = 100
        fc_vel_dict = {}
        for i in range(0, int((vel_stop-vel_start)/vel_step)):
            fc_vel_dict[str(i*10)+'-'+str((i+1)*10)] = {'dis_sum': 0, 'fc_sum': 0, 'fc_100_km':0}
        # end of settings--------------------------------------

        # variable initialization-------------------------------
        # cycle info (info element)
        cycle_index = ['EDU', '模式', '驾驶员', '日期', '开始时间', '里程(km)', '平均车速(kph)', '油耗(L)', '电耗(kwh)', '净油耗(L)',
                       '百公里油耗(L/100km)', '空调电耗(kwh)', '发动机工作时间(s)', '怠速时间(s)', '怠速充电时间(s)', '怠速时充电时间占比(%)',
                       '平均油门', '平均油门梯度(%/s)', '平均制动', '油门时间', '制动时间']
        cycle_result = {}
        for item in cycle_index:
            cycle_result[item] = []

        # operation map
        cycle_operation = []

        # hybrid mode
        hybrid_mode_list = ['默认模式', '怠速充电', '并联回收', '纯电驱动', '纯发动机', '串联驱动', '并联驱动', '串联回收', '电桩充电']
        hybrid_mode_color = ['crimson', 'limegreen', 'mediumorchid', 'orange', 'blue', 'olive', 'grey', 'cyan', 'black']
        cycle_hybrid_mode = []

        # statics info based on cycle info (integrated by edu type)
        statics_index_edu = ['试验数', '总里程(km)', '总油耗(L)', '总电耗(kwh)', '净油耗(L)', '单圈净油耗', '百公里净油耗(L/100km)',
                             '发动机启动时间(s)', '怠速时间(s)', '怠速充电时间(s)', '怠速时充电时间占比(%)']
        statics_group_edu = ['GEN1', 'GEN2']
        statics_result_edu = {}
        edu_operation = {}
        edu_hybrid_mode = {}
        for item in statics_group_edu:
            statics_result_edu[item] = {}
            edu_operation[item] = np.zeros(xx.shape)

        # statics info based on cycle info (integrated by edu type & driver)
        statics_index_driver = ['试验数', '总里程(km)', '总油耗(L)', '总电耗(kwh)', '净油耗(L)', '百公里净油耗(L/100km)', '平均油门', '平均油门梯度(%/s)', '平均制动']
        statics_group_driver = ['王京禹', '李凯', '姚利明']
        statics_result_driver = {}
        driver_operation = {}
        driver_hybrid_mode = {}
        for edu in statics_group_edu:
            statics_result_driver[edu] = {}
            driver_operation[edu] = {}
            driver_hybrid_mode[edu] = {}
            for driver in statics_group_driver:
                statics_result_driver[edu][driver] = {}
                driver_operation[edu][driver] = np.zeros(xx.shape)
                driver_hybrid_mode[edu][driver] = {}
        # end of variable initialization--------------------------------------------

        # extra input--------------------------------------
        input_file_path = r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2'
        input_file_name = r'GEN1_GEN2实际道路能耗试验记录.xlsx'
        input_file = os.path.join(input_file_path, input_file_name)
        data = xlrd.open_workbook(input_file)
        table = data.sheet_by_name('实验记录_有效数据')
        for i in range(2, 43):  # 43
            cycle_result['驾驶员'].append(table.cell(i, 0).value)
        for i in range(2, 43):
            cycle_result['驾驶员'].append(table.cell(i, 7).value)
        # end of extra input--------------------------------------

        # cyclic data extract and calculation--------------------------------------
        loop_number = 0
        for edu_type in self.range:
            filepath = os.path.join(root_path, edu_type)
            if test_mode:
                filepath = os.path.join(filepath, 'shell')
            filelist = os.listdir(filepath)

            for i in range(0, len(filelist)):
                loop_number += 1
                filename = filelist[i]
                filename_full = os.path.join(filepath, filename)
                info = filename.split('_')

                # if '.mdf' in filename:
                #     Data_ful = mdfreader.mdf(filename_full)
                #     Data_ful = Data_ful.convertToPandas(sampling=1 / fre)
                if '.csv' in filename:
                    Data_ful = pd.read_csv(filename_full, header=14, skiprows=[15, 16], skip_blank_lines=False, encoding='GB18030')
                else:
                    continue

                print('start processing:  ' + edu_type + '  ' + filename)

                # raw data
                fc_acc_raw = Data_ful[dbc[edu_type]['fc_acc']].tolist()  # uL
                vel_gps = Data_ful[dbc[edu_type]['velocity']].tolist()
                voltage = Data_ful[dbc[edu_type]['voltage']].tolist()
                current = Data_ful[dbc[edu_type]['current']].tolist()
                power = [voltage[i]*current[i] for i in range(0, len(current))]
                ac_power = Data_ful[dbc[edu_type]['ac_power']].tolist()
                pedal = Data_ful[dbc[edu_type]['pedal']].tolist()
                brake = Data_ful[dbc[edu_type]['brake']].tolist()
                enspd = Data_ful[dbc[edu_type]['enspd']].tolist()
                toq = Data_ful[dbc[edu_type]['toq']].tolist()
                hybrid_mode = Data_ful[dbc[edu_type]['hybrid_mode']].tolist()

                # instant fc calculation
                fc_acc_sub = [(fc_acc_raw[i]-fc_acc_raw[i-1]) for i in range(1, len(fc_acc_raw))]  # uL
                fc_acc_L = [x if x >= 0 else x+65520 for x in fc_acc_sub]  # uL
                fc_acc_L = [x/1000000 for x in fc_acc_L]  # L
                fc_acc_L.insert(0, 0)
                # data filtering: using interation to ensure continute error value removed
                fc_acc_L = [fc_acc_L[i] if fc_acc_L[i] < thre_value else fc_acc_L[i-1] for i in range(0, len(fc_acc_L))]

                # fc_acc_L_100km = [fc_acc_L[i]/(vel_gps[i]/3600/fre)*100 for i in range(0, len(fc_acc_L))]

                for j in range(0, int((vel_stop - vel_start) / vel_step)):
                    flag = [1 if (j+1)*10 > vel > j*10 else 0 for vel in vel_gps]
                    for k, l in groupby(enumerate(np.array(np.where(flag)[0])), lambda x: x[1] - x[0]):
                        index_list = [index for group, index in l]
                        length = index_list[-1] - index_list[0]
                        if length > fre*5:
                            start = index_list[0]
                            end = index_list[-1]
                            fc_vel_dict[str(j*10)+'-'+str((j+1)*10)]['fc_sum'] += sum(fc_acc_L[start:end+1])
                            fc_vel_dict[str(j * 10) + '-' + str((j + 1) * 10)]['dis_sum'] += sum(vel_gps[start:end+1])/3600/fre
                    try:
                        fc_vel_dict[str(j * 10) + '-' + str((j + 1) * 10)]['fc_100_km'] = \
                            fc_vel_dict[str(j*10)+'-'+str((j+1)*10)]['fc_sum']/fc_vel_dict[str(j * 10) + '-' + str((j + 1) * 10)]['dis_sum']*100
                    except ZeroDivisionError:
                        fc_vel_dict[str(j * 10) + '-' + str((j + 1) * 10)]['fc_100_km'] = 0

                # cycle info calculation
                cycle_result['日期'].append(info[0])
                cycle_result['开始时间'].append(info[1][0:6])
                cycle_result['EDU'].append(edu_type)
                cycle_result['模式'].append(round(np.mean(Data_ful[dbc[edu_type]['mode']].tolist()), 2))
                cycle_result['里程(km)'].append(round(sum(vel_gps)/3600/fre, 1))
                cycle_result['平均车速(kph)'].append(round(np.mean(vel_gps), 1))
                cycle_result['油耗(L)'].append(round(sum(fc_acc_L), 2))
                cycle_result['电耗(kwh)'].append(round(sum(power)/fre/1000/3600, 2))
                cycle_result['净油耗(L)'].append(round(cycle_result['油耗(L)'][-1] + cycle_result['电耗(kwh)'][-1]/2.22, 2))
                cycle_result['百公里油耗(L/100km)'].append(round(cycle_result['净油耗(L)'][-1]/cycle_result['里程(km)'][-1]*100, 2))
                cycle_result['空调电耗(kwh)'].append(round(sum(ac_power)/fre/3600, 2))
                cycle_result['平均油门'].append(round(np.mean([value for value in pedal if value > 0]), 1))
                cycle_result['平均油门梯度(%/s)'].append(round(np.mean([pedal[i+1] - pedal[i] for i in range(0, len(pedal)-1) if (pedal[i+1] - pedal[i]) > 0]), 1) * fre)
                cycle_result['平均制动'].append(round(np.mean([value for value in brake if value > 0]), 1))
                cycle_result['油门时间'].append(int(len([1 for value in pedal if value > 0])/fre))
                cycle_result['制动时间'].append(int(len([1 for value in brake if value > 0])/fre))
                cycle_result['发动机工作时间(s)'].append(int(sum([1 for i in range(0, len(enspd)) if (enspd[i] > 500) & (toq[i] > 0)])/fre))
                cycle_result['怠速时间(s)'].append(round(sum([1 for i in range(0, len(vel_gps)) if (enspd[i] > 500) & (vel_gps[i] == 0)])/fre, 1))
                cycle_result['怠速充电时间(s)'].append(round(sum([1 for i in range(0, len(vel_gps)) if (enspd[i] > 500) & (vel_gps[i] == 0) &
                                                            (current[i] < 0)])/fre, 1))
                if cycle_result['怠速时间(s)'][-1] != 0:
                    cycle_result['怠速时充电时间占比(%)'].append(round(cycle_result['怠速充电时间(s)'][-1]/cycle_result['怠速时间(s)'][-1], 2)*100)
                else:
                    cycle_result['怠速时充电时间占比(%)'].append(0)

                # cycle engine operation map
                cycle_operation.append(np.zeros(xx.shape))
                for j in range(0, len(enspd)):
                    if (enspd[j] > 500) & (toq[j] > 0):
                        cycle_operation[-1][int(toq[j] // mesh_grid_toq), int(enspd[j] // mesh_grid_enspd)] += 1
                cycle_operation[-1] = cycle_operation[-1] / fre

                fig = plt.pcolor(xx, yy, cycle_operation[-1], cmap='Reds')  # Reds
                fig.set_clim(vmin=0, vmax=120)
                plt.colorbar(fig)
                plt.title(cycle_result['EDU'][-1] + '\n' + '百公里净油耗: ' + str(cycle_result['百公里油耗(L/100km)'][-1]) + '\n' +
                          '发动机做功时间: ' + str(cycle_result['发动机工作时间(s)'][-1]))
                plt.grid(True)
                os.chdir(r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2\operation_points\cycle')
                if loop_number == 1:
                    clear_file_folder(r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2\operation_points\cycle')
                plt.savefig(cycle_result['EDU'][-1] + '_' + cycle_result['日期'][-1] + '_' + cycle_result['开始时间'][-1] + '_' +
                            str(cycle_result['发动机工作时间(s)'][-1]) + '.png', transparent=True)
                plt.close()

                # cycle hybrid mode distribution
                mode_distribution = []
                for j in range(0, 9):
                    mode_distribution.append(hybrid_mode.count(j))
                cycle_hybrid_mode.append(mode_distribution)
        # end of cyclic data extract and calculation--------------------------------------

        # settle length of extract input not equal with data record
        cycle_result['驾驶员'] = cycle_result['驾驶员'][0: len(cycle_result['EDU'])]
        cycle_result_pd = pd.DataFrame(cycle_result, columns=cycle_index)

        # edu based statics--------------------------------------
        loop_number = 0
        for group in statics_group_edu:
            loop_number += 1
            mask = cycle_result_pd['EDU'] == group
            statics_result_edu[group]['试验数'] = sum([1 for edu in cycle_result_pd['EDU'] if edu == group])
            statics_result_edu[group]['总里程(km)'] = round(sum(cycle_result_pd['里程(km)'][mask]), 1)
            statics_result_edu[group]['总油耗(L)'] = round(sum(cycle_result_pd['油耗(L)'][mask]), 2)
            statics_result_edu[group]['总电耗(kwh)'] = round(sum(cycle_result_pd['电耗(kwh)'][mask]), 2)
            statics_result_edu[group]['净油耗(L)'] = round(sum((cycle_result_pd['油耗(L)'] + cycle_result_pd['电耗(kwh)']/2.22)[mask]), 2)
            statics_result_edu[group]['单圈净油耗'] = round(statics_result_edu[group]['净油耗(L)']/statics_result_edu[group]['试验数'], 2)
            statics_result_edu[group]['百公里净油耗(L/100km)'] = round(statics_result_edu[group]['净油耗(L)']/statics_result_edu[group]['总里程(km)']
                                                                       * 100, 2)
            statics_result_edu[group]['发动机启动时间(s)'] = int(np.mean(cycle_result_pd['发动机工作时间(s)'][mask]))
            statics_result_edu[group]['怠速时间(s)'] = int(np.mean(cycle_result_pd['怠速时间(s)'][mask]))
            statics_result_edu[group]['怠速充电时间(s)'] = int(np.mean(cycle_result_pd['怠速充电时间(s)'][mask]))
            if statics_result_edu[group]['怠速时间(s)'] != 0:
                statics_result_edu[group]['怠速时充电时间占比(%)'] = round(statics_result_edu[group]['怠速充电时间(s)'] /
                                                                           statics_result_edu[group]['怠速时间(s)'] * 100, 2)
            else:
                statics_result_edu[group]['怠速时充电时间占比(%)'] = 0

            # draw operation map for edu
            fig_name = group + '_' + str(statics_result_edu[group]['发动机启动时间(s)']) + '.png'
            fig_title = group + '\n' + '百公里净油耗: ' + str(statics_result_edu[group]['百公里净油耗(L/100km)']) + '\n' + '发动机启动时间: ' + \
                        str(statics_result_edu[group]['发动机启动时间(s)'])
            save_folder = r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2\operation_points\edu'
            if loop_number == 1:
                clear_file_folder(save_folder)
            edu_operation[group] = draw_operation_map(mask=mask, cycle_operation=cycle_operation, xx=xx, yy=yy, fig_name=fig_name,
                                                      fig_title=fig_title, save_folder=save_folder)

            # draw hybrid mode distribution pie for edu
            save_folder = r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2\hybrid_mode\edu'
            fig_name = group + '_模式分布'
            fig_title = group
            if loop_number == 1:
                clear_file_folder(save_folder)
            edu_hybrid_mode[group] = draw_hybrid_mode_pie(mask=mask, cycle_hybrid_mode=cycle_hybrid_mode, label_list=hybrid_mode_list,
                                                          color_list=hybrid_mode_color, fig_title=fig_title, save_folder=save_folder, fig_name=fig_name)

        statics_result_edu_pd = pd.DataFrame(statics_result_edu, columns=statics_group_edu, index=statics_index_edu)
        # end of edu based statics--------------------------------------

        # edu-driver based statics--------------------------------------
        statics_result_driver_pd = {}
        loop_number = 0
        for edu in statics_group_edu:
            for driver in statics_group_driver:
                if driver in cycle_result['驾驶员']:
                    loop_number += 1
                    mask = (cycle_result_pd['EDU'] == edu) & (cycle_result_pd['驾驶员'] == driver)
                    statics_result_driver[edu][driver]['试验数'] = sum([1 for i in range(0, len(cycle_result_pd['EDU']))
                                                                     if (cycle_result_pd['EDU'][i] == edu) & (cycle_result_pd['驾驶员'][i] == driver)])
                    statics_result_driver[edu][driver]['总里程(km)'] = round(sum(cycle_result_pd['里程(km)'][mask]), 1)
                    statics_result_driver[edu][driver]['总油耗(L)'] = round(sum(cycle_result_pd['油耗(L)'][mask]), 2)
                    statics_result_driver[edu][driver]['总电耗(kwh)'] = round(sum(cycle_result_pd['电耗(kwh)'][mask]), 2)
                    statics_result_driver[edu][driver]['净油耗(L)'] = round(sum((cycle_result_pd['油耗(L)'] + cycle_result_pd['电耗(kwh)']/2.22)[mask]), 2)
                    if statics_result_driver[edu][driver]['总里程(km)'] > 0:
                        statics_result_driver[edu][driver]['百公里净油耗(L/100km)'] = round(statics_result_driver[edu][driver]['净油耗(L)']/statics_result_driver[edu][driver]['总里程(km)']*100, 2)
                        statics_result_driver[edu][driver]['平均油门'] = round(sum(cycle_result_pd['平均油门'][mask] * cycle_result_pd['油门时间'][mask]) / sum(cycle_result_pd['油门时间'][mask]), 1)
                        statics_result_driver[edu][driver]['平均制动'] = round(sum(cycle_result_pd['平均制动'][mask] * cycle_result_pd['制动时间'][mask]) / sum(cycle_result_pd['制动时间'][mask]), 1)
                    else:
                        statics_result_driver[edu][driver]['百公里净油耗(L/100km)'] = 0
                        statics_result_driver[edu][driver]['平均油门'] = 0
                        statics_result_driver[edu][driver]['平均制动'] = 0
                        statics_result_driver[edu][driver]['平均油门梯度(%/s)'] = round(np.mean(cycle_result_pd['平均油门梯度(%/s)'][mask]))

                    # draw operation map for driver
                    fig_name = edu + '_' + driver + '_' + str(statics_result_driver[edu][driver]['百公里净油耗(L/100km)']) + '.png'
                    fig_title = edu + '_' + driver + '\n' + '百公里净油耗: ' + str(statics_result_driver[edu][driver]['百公里净油耗(L/100km)'])
                    save_folder = r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2\operation_points\driver'
                    if loop_number == 1:
                        clear_file_folder(save_folder)
                    driver_operation[edu][driver] = draw_operation_map(mask=mask, cycle_operation=cycle_operation, xx=xx, yy=yy, fig_name=fig_name,
                                                                       fig_title=fig_title, save_folder=save_folder)
                    # draw hybrid mode distribution pie for dirver
                    save_folder = r'D:\pythonCodes\function_stlye_module\fuel consumption\real_road_GEN1_GEN2\hybrid_mode\driver'
                    fig_name = edu + '_' + driver + '_模式分布'
                    fig_title = group + '_' + driver
                    if loop_number == 1:
                        clear_file_folder(save_folder)
                    driver_hybrid_mode[edu][driver] = draw_hybrid_mode_pie(mask=mask, cycle_hybrid_mode=cycle_hybrid_mode, fig_title=fig_title,
                                                                           label_list=hybrid_mode_list, color_list=hybrid_mode_color,
                                                                           save_folder=save_folder, fig_name=fig_name)

            statics_result_driver_pd[edu] = pd.DataFrame(statics_result_driver[edu], columns=statics_group_driver, index=statics_index_driver)
        # end of edu-driver based statics--------------------------------------

        # result output--------------------------------------
        writer = pd.ExcelWriter(os.path.join(self.filepath, 'result.xlsx'))
        cycle_result_pd.to_excel(writer, sheet_name='cycle result', startrow=0, startcol=0)
        statics_result_edu_pd.to_excel(writer, sheet_name='EDU-based statics result', startrow=0, startcol=0)
        j = 0
        for edu in statics_group_edu:
            statics_result_driver_pd[edu].to_excel(writer, sheet_name='EDU-Driver-based statics result', startrow=0, startcol=j)
            j = j + len(statics_group_driver) + 2
        writer.save()
        # end of result output--------------------------------------


if __name__ == '__main__':
    test_mode = False
    work_range = ['GEN1', 'GEN2']
    root_path = 'D:/pythonCodes/function_stlye_module/fuel consumption/real_road_GEN1_GEN2'
    dbc_dict = {'GEN1':
                       {'velocity': 'VehSpdAvgNonDrvn_h1HSC1',
                        'fc_acc': 'FuelCsumpHSC1',
                        'voltage': 'LSBMSPackVol_h6HSC6',
                        'current': 'LSBMSPackCrnt_h6HSC6',
                        'mode': 'EPTDrvngMdSwStsHSC1',
                        'date': '日期',
                        'time': '时间',
                        'ac_power': 'ACComprActuPwrHSC1',
                        'pedal': 'EPTAccelActuPosHSC1',
                        'brake': 'BrkPdlDrvrAppdPrs_h1HSC1',
                        'enspd': 'EnSpdHSC1',
                        'toq': 'EnActuStdyStaToqHSC1',
                        'hybrid_mode': 'ElecVehSysMdHSC1',
                        },
                'GEN2':
                       {'velocity': 'VehSpdAvgNonDrvnHSC1',
                        'fc_acc': 'FuelCsumpHSC1',
                        'voltage': 'BMSPackVol_h1HSC1',
                        'current': 'BMSPackCrnt_h1HSC1',
                        'mode': 'EPTDrvngMdSwStsHSC1',
                        'date': '日期',
                        'time': '时间',
                        'ac_power': 'ACComprActuPwrHSC1',
                        'pedal': 'EPTAccelActuPosHSC1',
                        'brake': 'BrkPdlDrvrAppdPrs_h1HSC1',
                        'enspd': 'EnSpdHSC1',
                        'toq': 'EnActuStdyStaToqHSC1',
                        'hybrid_mode': 'ElecVehSysMdHSC1',
                        }
                }
    a = FuelConsumption(root_path=root_path, work_range=work_range, dbc_dict=dbc_dict, test_mode=test_mode)
    a.fc_main()
    print('Finish!')
