#!/usr/bin/env python
# -*- coding:utf-8 -*-

import mdfreader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xlwt
import xlsxwriter


class FuelConsumption(object):

    def __init__(self):
        self.filepath = root_path
        self.range = work_range
        self.dbc = dbc_dict

    def fc_main(self):
        fre = 20
        thre_value = 0.5
        mesh_grid_enspd = 200
        mesh_grid_toq = 10
        dbc = self.dbc

        # summary info
        date = []
        start_time = []
        fc = []  # fuel consumption in L
        ec = []  # electric power consumption in kw
        dis = []  # distance in km
        fe = []  # fuel economic in L/100km
        mode = []  # drive mode
        vel_avg = []  # velocity average in kph
        edu = []  # edu type
        ac = []  # AC power consumption in kwh
        ped_avg = []  # average pedal in %
        brk_avg = []  # average brake pressure in kpa

        # statics info from summary
        target = []
        cycle_tot = []
        dis_tot = []
        fc_tot = []
        ec_tot = []
        nfc_tot = []
        nfc_cycle = []
        nfe = []
        runtime_en_ave = []

        runtime_en = []  # engine speed in rpm
        style = xlwt.XFStyle()  # Create Style
        style.alignment.horz = 2  # 字体居中

        for edu_type in self.range:
            filepath = os.path.join(root_path, edu_type)
            filelist = os.listdir(filepath)

            for i in range(0, len(filelist)):
                filename = filelist[i]
                filename_full = os.path.join(filepath, filename)

                # if '.mdf' in filename:
                #     Data_ful = mdfreader.mdf(filename_full)
                #     Data_ful = Data_ful.convertToPandas(sampling=1 / fre)
                if '.csv' in filename:
                    Data_ful = pd.read_csv(filename_full, header=14, skiprows=[15, 16], skip_blank_lines=False, encoding='GB18030')
                else:
                    continue

                print('start processing:  ' + edu_type + '  ' + filename)
                info = filename.split('_')
                date.append(info[0])
                start_time.append(info[1][0:6])
                edu.append(edu_type)

                # raw data
                fc_acc_raw = Data_ful[dbc[edu_type]['fc_acc']].tolist()  # uL
                vel_gps = Data_ful[dbc[edu_type]['velocity']].tolist()  #
                voltage = Data_ful[dbc[edu_type]['voltage']].tolist()
                current = Data_ful[dbc[edu_type]['current']].tolist()
                power = [voltage[i]*current[i] for i in range(0, len(current))]
                ac_power = Data_ful[dbc[edu_type]['ac_power']].tolist()
                pedal = Data_ful[dbc[edu_type]['pedal']].tolist()
                brake = Data_ful[dbc[edu_type]['brake']].tolist()
                enspd = Data_ful[dbc[edu_type]['enspd']].tolist()
                toq = Data_ful[dbc[edu_type]['toq']].tolist()

                # instant fc calculation
                fc_acc_sub = [(fc_acc_raw[i]-fc_acc_raw[i-1]) for i in range(1, len(fc_acc_raw))]  # uL
                fc_acc_L = [x if x >= 0 else x+65520 for x in fc_acc_sub]  # uL
                fc_acc_L = [x/1000000 for x in fc_acc_L]  # L
                fc_acc_L.insert(0, 0)

                # data filtering: using interation to ensure continute error value removed
                fc_acc_L = [fc_acc_L[i] if fc_acc_L[i] < thre_value else fc_acc_L[i-1] for i in range(0, len(fc_acc_L))]

                mode.append(round(np.mean(Data_ful[dbc[edu_type]['mode']].tolist()), 2))
                dis.append(round(sum(vel_gps)/3600/fre, 2))  # km
                vel_avg.append(round(np.mean(vel_gps), 2))
                fc.append(round(sum(fc_acc_L), 2))
                fe.append(round(fc[-1]/dis[-1]*100, 2))  # L/100km
                ec.append(round(sum(power)/fre/1000/3600, 2))
                ac.append(round(sum(ac_power)/fre/3600, 2))
                ped_avg.append(round(np.mean(pedal)))
                brk_avg.append(round(np.mean(brake)))
                runtime_en.append(sum([1/fre for x in enspd if x > 500]))  # time of engine run

            mask = [1 if x == edu_type else 0 for x in edu]
            target.append(edu_type)
            cycle_tot.append(edu.count(edu_type))
            dis_tot.append(sum([dis[i] for i in range(0, len(dis)) if mask[i] == 1]))
            fc_tot.append(sum([fc[i] for i in range(0, len(fc)) if mask[i] == 1]))
            ec_tot.append(sum([ec[i] for i in range(0, len(ec)) if mask[i] == 1]))
            nfc_tot.append(round(sum([fc[i]+ec[i]/2.22 for i in range(0, len(fc)) if mask[i] == 1]), 2))
            nfc_cycle.append(round(nfc_tot[-1]/cycle_tot[-1], 2))
            nfe.append(round(nfc_tot[-1]/dis_tot[-1]*100, 2))
            runtime_en_ave.append(int(np.mean([runtime_en[i] for i in range(0, len(runtime_en)) if mask[i] == 1])))

        summary_title = ['EDU', '日期', '开始时间', '里程(km)', '平均车速(kph)', '模式', '油耗(L)', '电耗(kwh)', '百公里油耗(L/100km)', '空调电耗',
                         '发动机工作时间', '平均踏板', '平均制动']
        summary_value = [edu, date, start_time, dis, vel_avg, mode, fc, ec, fe, ac, runtime_en, ped_avg, brk_avg]
        book = xlwt.Workbook(encoding='GB18030', style_compression=0)
        sheet = book.add_sheet('summary', cell_overwrite_ok=True)

        for i in range(0, len(summary_title)):
            sheet.write(0, i, summary_title[i], style)
            for j in range(0, len(date)):
                sheet.write(j+1, i, summary_value[i][j], style)

        statics_title = ['对象', '试验数', '总里程(km)', '总油耗(L)', '总电耗(kwh)', '净油耗(L)', '单圈净油耗', '百公里净油耗(L/100km)', '平均发动机启动时间(s)']
        statics_value = [target, cycle_tot, dis_tot, fc_tot, ec_tot, nfc_tot, nfc_cycle, nfe, runtime_en_ave]
        for k in range(0, len(statics_title)):
            sheet.write(k, i+3, statics_title[k], style)
            for l in range(0, len(cycle_tot)):
                sheet.write(k, i+4+l, statics_value[k][l], style)

        compare_title = ['日期', '开始时间', '里程(km)', '平均车速(kph)', '油耗(L)', '电耗(kwh)', '百公里油耗(L/100km)', '空调电耗',
                         '发动机工作时间']
        compare_value = [date, start_time, dis, vel_avg, fc, ec, fe, ac, runtime_en]
        sheet1 = book.add_sheet('compare')

        for i in range(0, len(compare_title)):
            sheet1.write_merge(0, 0, 2*i, 2*i+1, compare_title[i], style)
            sheet1.write(1, 2*i, target[0])
            sheet1.write(1, 2*i+1, target[1])
            for j in range(0, min(cycle_tot)):
                sheet1.write(j+2, 2*i, compare_value[i][j], style)
                sheet1.write(j+2, 2*i+1, compare_value[i][j+min(cycle_tot)], style)

        book.save(self.filepath + '\summary.xls')


if __name__ == '__main__':
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
                'toq': 'EnActuStdyStaToqHSC1'},
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
                'toq': 'EnActuStdyStaToqHSC1'
                }
           }
    a = FuelConsumption(root_path=root_path, work_range=work_range, dbc_dict=dbc_dict)
    a.fc_main()
    print('Finish!')
