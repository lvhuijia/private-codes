#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xlwt
import xlsxwriter


class FuelConsumption(object):


    def __init__(self, **kwargs):
        """
        Initial function of system gain.

        :param file_path: path of system gain file in local disc
        """
        self.filepath = kwargs['file_path']

    def fc_main(self):
        '''
        Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.

        '''
        filelist = os.listdir(self.filepath)
        fre = 20
        thre_value = 0.0005
        thre_start = 10*fre
        number = []
        date = []
        driver = []
        mode = []
        fc_ins_nondri_raw = []
        fc_acc_dri_raw = []
        fc_ins_nondri_process = []
        fc_acc_dri_process = []

        for i in range(0, len(filelist)):
            filename = filelist[i]
            info = filename.split('_')
            number.append(int(info[0]))
            date.append(info[1])
            driver.append(info[2])
            mode.append(info[3])
            filename = self.filepath + '/' + filename
            fga_Data_ful = pd.read_csv(filename, encoding='GB18030')
            fga_Data_Selc = fga_Data_ful.loc[:, ['CCP_vsksml_w', 'FuelCsumpHSC1', 'VehSpdAvgDrvnHSC1', 'VehSpdAvgNonDrvnHSC1']]

            # raw data
            fc_ins_raw = fga_Data_Selc['CCP_vsksml_w'].tolist()  # mL/s
            fc_acc_raw = fga_Data_Selc['FuelCsumpHSC1'].tolist()  # uL
            vel_dri = fga_Data_Selc['VehSpdAvgDrvnHSC1'].tolist()  # kph
            vel_nondri = fga_Data_Selc['VehSpdAvgNonDrvnHSC1'].tolist()  # kph

            # instant fc calculation
            fc_ins_L = [x/1000/fre for x in fc_ins_raw]  # L
            fc_acc_sub = [(fc_acc_raw[i]-fc_acc_raw[i-1]) for i in range(1, len(fc_acc_raw))]  # uL
            fc_acc_L = [x if x >= 0 else x+65520 for x in fc_acc_sub]  # uL
            fc_acc_L = [x/1000000 for x in fc_acc_L]  # L
            fc_acc_L.insert(0, 0)

            # data cutting
            fc_ins_L_process = [fc_ins_L[i] if i >= thre_start else 0 for i in range(0, len(fc_ins_L))]
            fc_acc_L_process = [fc_acc_L[i] if i >= thre_start else 0 for i in range(0, len(fc_acc_L))]
            vel_dri_process = [vel_dri[i] if i >= thre_start else 0 for i in range(0, len(vel_dri))]  # km
            vel_nondri_process = [vel_nondri[i] if i >= thre_start else 0 for i in range(0, len(vel_nondri))]  # km

            # data filtering: using interation to ensure continute error value removed
            # fc_ins_L_process = [fc_ins_L_process[i] if fc_ins_L_process[i] < thre_value else fc_ins_L_process[i-1] for i in range(0, len(fc_ins_L_process))]
            # fc_acc_L_process = [fc_acc_L_process[i] if fc_acc_L_process[i] < thre_value else fc_acc_L_process[i-1] for i in range(0, len(fc_acc_L_process))]
            for j in range(0, len(fc_ins_L_process)):
                if fc_ins_L_process[j] > thre_value and j >= 0:
                    fc_ins_L_process[j] = fc_ins_L_process[j-1]

            for j in range(0, len(fc_acc_L_process)):
                if fc_acc_L_process[j] > thre_value and j >= 0:
                    fc_acc_L_process[j] = fc_acc_L_process[j-1]

            dis_dri = sum(vel_dri)/3600/fre  # km
            dis_nondri = sum(vel_nondri)/3600/fre  # km
            fc_ins_nondri_raw.append(sum(fc_ins_L)/dis_nondri*100)  # L/100km
            fc_acc_dri_raw.append(sum(fc_acc_L)/dis_dri*100)  # L/100km
            fc_ins_nondri_process.append(sum(fc_ins_L_process)/dis_nondri*100)  # L/100km
            fc_acc_dri_process.append(sum(fc_acc_L_process)/dis_dri*100)  # L/100km

            # data output
            os.chdir(r'D:\pythonCodes\function_stlye_module\fuel consumption\result_real_road')
            book = xlsxwriter.Workbook(filelist[i].split('.')[0] + '.xlsx')
            sheet = book.add_worksheet('procedure data')
            sheet.write(0, 0, 'fc_ins_L_raw')
            sheet.write(0, 1, 'fc_acc_L_raw')
            sheet.write(0, 2, 'fc_ins_L_process')
            sheet.write(0, 3, 'fc_acc_L_process')
            sheet.write(0, 4, 'fc_ins_nondri')
            sheet.write(0, 5, 'fc_acc_dri')
            sheet.write(2, 4, fc_ins_nondri_process[i])
            sheet.write(2, 5, fc_acc_dri_process[i])
            sheet.write(1, 4, fc_ins_nondri_raw[i])
            sheet.write(1, 5, fc_acc_dri_raw[i])
            sheet.write(1, 6, 'raw')
            sheet.write(2, 6, 'process')

            for j in range(0, len(fc_ins_L)):
                sheet.write(j+1, 0, fc_ins_L[j])
                sheet.write(j+1, 1, fc_acc_L[j])
                sheet.write(j+1, 2, fc_ins_L_process[j])
                sheet.write(j+1, 3, fc_acc_L_process[j])

            # book.save('D:/pythonCodes/function_stlye_module/fuel consumption/result/'+filelist[i][0:-4]+'.xlsx')
            book.close()

        book = xlwt.Workbook(encoding='GB18030', style_compression=0)
        sheet = book.add_sheet('summary', cell_overwrite_ok=True)
        sheet.write(0, 0, '编号')
        sheet.write(0, 1, '日期')
        sheet.write(0, 2, '驾驶员')
        sheet.write(0, 3, '模式')
        sheet.write(0, 4, '累积油耗/主动轮')
        sheet.write(0, 5, '瞬时油耗/从动轮')

        for i in range(0, len(filelist)):
            sheet.write(number[i], 0, number[i])
            sheet.write(number[i], 1, date[i])
            sheet.write(number[i], 2, driver[i])
            sheet.write(number[i], 3, mode[i])
            sheet.write(number[i], 4, fc_acc_dri_process[i])
            sheet.write(number[i], 5, fc_ins_nondri_process[i])
        book.save(r'D:\pythonCodes\function_stlye_module\fuel consumption\summary.xls')


if __name__ == '__main__':
    # *******1-GetSysGainData****** AS22_C16UVV016_SystemGain_20160925_D_M_SL, IP31_L16UOV055_10T_SystemGain_20160225
    a = FuelConsumption(file_path='D:/pythonCodes/function_stlye_module/fuel consumption/real_road')
    a.fc_main()
    # plt.show()
    print('Finish!')
