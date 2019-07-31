#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt
import pywt
from mongo import MongoClient
import datetime


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
        self.filepath = kwargs['file_path']
        self.tipin_class = {}

    def tipin_main(self):
        '''
        Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.

        '''
        # parameters setting
        fre = 20  # sample rate
        velocity_max_still = 0.1  # maximum velocity when stay still, used for acc offset correction
        # parameters setting ends

        tipin_Data_ful = pd.read_csv(self.filepath)
        tipin_Data_Selc = tipin_Data_ful.loc[:, ['AccelActuPosHSC1', 'EnSpdHSC1', 'MSLongAccelGHSC',
                                                 'VehSpdAvgNonDrvnHSC1', 'TrEstdGear_TCMHSC1', 'kickdown']]

        pedal_data = tipin_Data_Selc['AccelActuPosHSC1'].tolist()
        speed_data = tipin_Data_Selc['EnSpdHSC1'].tolist()
        acc_data = tipin_Data_Selc['MSLongAccelGHSC'].tolist()
        gear_data = tipin_Data_Selc['TrEstdGear_TCMHSC1'].tolist()
        velocity_data = tipin_Data_Selc['VehSpdAvgNonDrvnHSC1'].tolist()
        kickdown_data = tipin_Data_Selc['kickdown'].tolist()
        pedal_data = [120 if kickdown_data[i] == 1 else pedal_data[i] for i in range(0, len(pedal_data))]
        abstime_data = [i/fre for i in range(0, len(pedal_data))]

        # data import to MongoDB
        sum = 0
        num = 0
        mongo = MongoClient(dbName='IP40', collection='tip in data')
        for i in range(0, len(pedal_data)):
            rowdata = {
                "pedal": pedal_data[i],
                'speed': speed_data[i],
                "acc": acc_data[i],
                "velocity": velocity_data[i],
            }
            time_before = datetime.datetime.now()
            mongo.insert(rowdata)
            time_after = datetime.datetime.now()
            time_delta = time_after - time_before
            sum = sum + time_delta.microseconds
            num = num + 1
            aver = sum/num
            print(aver)
        print("Inserted new mongo record")
        # data import to MongoDB ends

        gear_max = int(max(gear_data))
        if gear_max > 9:
            gear_data_forward = [x for x in gear_data if x < 10]
            gear_max = int(max(gear_data_forward))

        # acc signal pre-processing
        # offset correct, error value removed undo
        still_flag = (np.array(velocity_data) < velocity_max_still) & (np.array(pedal_data) == 0) & (np.array(acc_data) < 1)
        acc_offset = sum(np.array(acc_data) * still_flag)/sum(still_flag)
        acc_correct = [x-acc_offset for x in acc_data]

        # wavelet decomposition & de-noising：de-noising not found
        acc_smooth = acc_correct
        # c = pywt.wavedec(acc_correct, wavelet='db4', mode='symmetric', level=2)
        # pywt.threshold(acc_correct, 2, 'soft', 6)
        # 'sym' is the default mode of matlab. db4 same with matlab, dmey not, dmey fits better

        pedal_var = [pedal_data[i+1]-pedal_data[i] for i in range(0, len(pedal_data)-1)]  # pedal variation at each time step
        pedal_var.insert(0, 0)  # 0 at the beginning
        pedal_rising_index = [i-1 for i in range(0, len(pedal_var)) if pedal_var[i] > 1]
        pedal_rising = [pedal_rising_index[i+1] for i in range(0, len(pedal_rising_index)-1) if (pedal_rising_index[i+1] - pedal_rising_index[i]) > 0.5*fre]  # index of pedal variation at next time step larger than 1%
        pedal_rising.insert(0, pedal_rising_index[0])
        pedal_rising.append(pedal_rising_index[-1])

        pedal_rising_left = pedal_rising[0:-1]  # start of a test(index)
        pedal_rising_right = [x - 1 for x in pedal_rising[1:]] # one time step before next start i.e. end of a test(index)
        pedal_rising_right[-1] = len(pedal_data)-1  # after the last test, no new test is started, so the end of the last test is set to be the end of the recording

        start = [pedal_rising_left[i] for i in range(0, len(pedal_rising_left)) if (pedal_rising_right[i] - pedal_rising_left[i]) >= 3*fre]  # time between two rising edges less than 3s is eliminated
        end = [pedal_rising_right[i] for i in range(0, len(pedal_rising_left)) if (pedal_rising_right[i] - pedal_rising_left[i]) >= 3*fre]

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
            pedal_max_process = [x + start[i] for x in range(0, len(pedal_cal)) if abs(pedal_cal[x] - pedal_max[i]) < 1.5] # process near the target pedal(global index)
            pedal_max_start.append(pedal_max_process[0])
            pedal_max_end.append(pedal_max_process[-1])

            pedal_min = min(pedal_data[pedal_max_end[i]:end[i]+1])
            pedal_min_process = [x for x in range(pedal_max_end[i], end[i]+1) if abs(pedal_data[x] - pedal_min) < 1.5] # process near the min pedal(global index)
            pedal_min_start.append(pedal_min_process[0])
            pedal_min_end.append(pedal_min_process[-1])

            acc_max_3s.append(max(acc_correct[start[i]:start[i] + 3*fre+1]))
            pedal_rise.append(pedal_max[i] - pedal_cal[0])
        # index calculation for data filtering ends

        # parameters setting: data filtering
        pedal_constantspeed_min = 0
        holdtime_min = 3  # minimum holding time of pedal
        acc_respond_min = 0.05  # minimum acc response in 3s
        pedal_rise_min = 1
        pedal_rise_time_max = 0.5
        # parameters setting ends

        # data filtering
        segement_raw = len(start)
        exclude_flag1 = [1 if pedal_data[start[i]] < pedal_constantspeed_min else 0 for i in range(0, segement_raw)]
        exclude_flag2 = [1 if acc_max_3s[i] < acc_respond_min else 0 for i in range(0, segement_raw)]
        exclude_flag3 = [1 if pedal_rise[i] <= pedal_rise_min else 0 for i in range(0, segement_raw)]
        exclude_flag4 = [1 if (pedal_max_start[i] - start[i]) > (pedal_rise_time_max*fre) else 0 for i in range(0, segement_raw)]
        exclude_flag5 = [1 if (pedal_max_end[i] - pedal_max_start[i]) < (holdtime_min*fre) else 0 for i in range(0, segement_raw)]
        exclude_flag = [max(exclude_flag1[i], exclude_flag2[i], exclude_flag3[i], exclude_flag4[i], exclude_flag5[i]) for i in range(0, segement_raw)]

        start = [start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        end = [end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        acc_max_3s = [acc_max_3s[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_rise = [pedal_rise[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_min_start = [pedal_min_start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_min_end = [pedal_min_end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max_start = [pedal_max_start[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max_end = [pedal_max_end[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        pedal_max = [pedal_max[i] for i in range(0, segement_raw) if exclude_flag[i] == 0]
        segement_filtered = len(start)
        # data filtering end

        # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        # sheet = book.add_sheet('test', cell_overwrite_ok=True)
        # for i in range(0, len(start)):
        #     sheet.write(i, 0, start[i])
        #     sheet.write(i, 1, end[i])
        #     sheet.write(i, 2, pedal_max_start[i])
        #     sheet.write(i, 3, acc_max_3s[i])
        #     sheet.write(i, 4, pedal_rise[i])
        #     sheet.write(i, 5, pedal_min_start[i])
        #     sheet.write(i, 6, pedal_min_end[i])
        #     sheet.write(i, 7, pedal_max_end[i])
        #     sheet.write(i, 8, pedal_max[i])
        # book.save(r'D:\pythonCodes\function_stlye_module\tip in\test.xls')

        # data cutting for each segement
        abstime_segement = [abstime_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        velocity_segement = [velocity_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        acc_segement = [acc_correct[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        pedal_segement = [pedal_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        acc_smooth_segement = [acc_smooth[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        gear_segement = [gear_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        speed_segement = [speed_data[start[i]:pedal_min_end[i]+1] for i in range(0, segement_filtered)]
        # data cutting ends

        # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        # for j in range(0, segement_filtered):
        #     sheet = book.add_sheet(str(j+1), cell_overwrite_ok=True)
        #     for i in range(0, len(abstime_segement[j])):
        #         sheet.write(i, 0, abstime_segement[j][i])
        #         sheet.write(i, 1, velocity_segement[j][i])
        #         sheet.write(i, 2, acc_segement[j][i])
        #         sheet.write(i, 3, pedal_segement[j][i])
        #         sheet.write(i, 4, acc_smooth_segement[j][i])
        #         sheet.write(i, 5, float(gear_segement[j][i]))
        #         sheet.write(i, 6, speed_segement[j][i])
        # book.save(r'D:\pythonCodes\function_stlye_module\tip in\test.xls')

        # evaluation index calculation.
        # parameters setting: for index calculation and plotting
        acc_delay_min = 0.05
        response_time_thre = 3
        time_plot = 6
        # parameters setting ends

        delay_index = []
        delay_time = []
        response_index = []
        response_time = []
        zeropointone_index = []
        zeropointone_time = []
        thump_rate = []
        kick = []
        curve_element = {}

        for i in range(0, segement_filtered):
            abstime_cal = abstime_segement[i]
            velocity_cal = velocity_segement[i]
            acc_cal = acc_smooth_segement[i]
            pedal_cal = pedal_segement[i]
            speed_cal = speed_segement[i]
            gear_cal = gear_segement[i]

            velocity_target = round(velocity_cal[0]/10)*10
            pedal_target = round(max(pedal_cal)/10)*10
            if (velocity_target in [20, 40, 60, 80, 100]) & (pedal_target in [40, 60, 80, 100, 120]):
                name = str(int(velocity_target)) + 'kph' + str(int(pedal_target)) + '%'
                if ~(name in curve_element.keys()) or ((name in curve_element.keys()) and (len(acc_cal) > len(curve_element[name]['acc']))):
                    curve_element[name] = {}
                    curve_element[name]['acc'] = acc_cal[0:time_plot*fre+1]
                    curve_element[name]['pedal'] = pedal_cal[0:time_plot*fre+1]
                    curve_element[name]['gear'] = gear_cal[0:time_plot*fre+1]
                    curve_element[name]['pedal_target'] = pedal_target
                    curve_element[name]['velocity_target'] = velocity_target

            # index calculation-------------------------------------------------------------------------------
            # delay
            for j in range(0, len(acc_cal)):
                if acc_cal[j] >= acc_delay_min:
                    delay_index.append(j)
                    break
            delay_time.append(abstime_cal[delay_index[i]] - abstime_cal[0])

            # response time
            acc_response = 0.95*max(acc_cal[0:response_time_thre*fre+1])
            for j in range(0, response_time_thre*fre+1):
                if acc_cal[j] >= acc_response:
                    response_index.append(j)
                    break
            response_time.append(abstime_cal[response_index[i]] - abstime_cal[0])

            # 0.1g time
            if max(acc_cal) >= 0.1:
                for j in range(0, len(acc_cal)):
                    if acc_cal[j] >= 0.1:
                        zeropointone_index.append(j)
                        zeropointone_time.append(abstime_cal[zeropointone_index[i]] - abstime_cal[0])
                        break
            else:
                zeropointone_index.append(-1)
                zeropointone_time.append(-1)

            # thump rate,start point TBD: maybe maximum acc change point
            if response_time[i] - delay_time[i] == 0:
                thump_rate.append(-1)
            else:
                thump_rate.append((acc_cal[response_index[i]] - acc_cal[delay_index[i]])/(response_time[i] - delay_time[i]))

            # kick calculation: TBD, needs better way to find peaks rather than simply using its definition for a wavy signal

            # geardown: input not specific enough

            # end of index calculation-------------------------------------------------------------------------------

        # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        # sheet = book.add_sheet('sheet', cell_overwrite_ok=True)
        # for i in range(0, segement_filtered):
        #     sheet.write(0, i, delay_time[i])
        #     sheet.write(1, i, response_time[i])
        #     sheet.write(2, i, zeropointone_time[i])
        #     sheet.write(3, i, thump_rate[i])
        # book.save(r'D:\pythonCodes\function_stlye_module\tip in\test.xls')

        TipIn.plot_acc_curve(self, curve_element, fre, time_plot, gear_max)

    def plot_acc_curve(self, curve_element, fre, time_plot, gear_max):
        self.fig1 = plt.figure()
        gear_flag = 1  # 1:with gear curve,0: without gear curve

        # colormap
        colormap = []
        colormap.append([0.8431, 0.6781, 0.2941])
        colormap.append([0.9061, 0.5491, 0.2241])
        colormap.append([0.7291, 0.4351, 0.2351])
        colormap.append([0.5181, 0.2631, 0.1531])
        colormap.append([0.3101, 0.3181, 0.1651])
        color_index_list = []

        for key in curve_element:
            # color & position determination
            color_index = int(curve_element[key]['pedal_target']/20)-2
            color_index_list.append(color_index)
            subfigure_index = int(curve_element[key]['velocity_target']/20)
            curve_element[key]['pedal'] = [100 if curve_element[key]['pedal'][i] == 120 else
                                           curve_element[key]['pedal'][i] for i in range(0, len(curve_element[key]['pedal']))]

            # acc curves
            ax1 = self.fig1.add_subplot(2, 3, subfigure_index)
            ax1.plot([i/fre for i in range(0, time_plot*fre+1)], curve_element[key]['acc'], color=colormap[color_index], linewidth=2)
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Acc (g)', fontsize=10)
            ax1.set_title(str(int(curve_element[key]['velocity_target'])), fontsize=12)
            ax1.set_xlim(0, time_plot)

            # pedal curves
            ax2 = ax1.twinx()
            ax2.plot([i/fre for i in range(0, time_plot*fre+1)], curve_element[key]['pedal'], color=colormap[color_index])
            ax2.set_ylim(0, 101)

            # gear curves if requested, setting the second y axis label accordingly
            if gear_flag == 1:
                ax3 = ax1.twinx()
                ax3.plot([i/fre for i in range(0, time_plot*fre+1)], curve_element[key]['gear'], ':', color=colormap[color_index])
                ax3.set_ylim(-gear_max - 1, gear_max + 1)
                ax3.set_yticks([i for i in range(0, gear_max + 2)])
                ax3.set_ylabel('Gear', fontsize=10)
                ax2.set_yticks([])
                ax1.set_ylim(0, 1)
                ax1.set_yticks([i * 0.1 for i in range(0, 6)])
            else:
                ax2.set_yticks([40+i*20 for i in range(0, 4)])
                ax2.set_ylabel('Pedal (%)', fontsize=10)
                ax1.set_ylim(0, 0.5)
                ax1.set_yticks([i * 0.1 for i in range(0, 6)])

        # legend add in seperate figure window：by adding a blank plot
        legend_pool = ['40%', '60%', '80%', '100%', 'kickdown']
        ax = self.fig1.add_subplot(2, 3, 6)
        for i in range(0, max(color_index_list)+1):
            ax.plot([], [], color=colormap[i])
        ax.legend(legend_pool[0:max(color_index_list)+1], loc=2, fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        plt.show()


if __name__ == '__main__':
    # *******1-GetSysGainData****** AS22_C16UVV016_SystemGain_20160925_D_M_SL, IP31_L16UOV055_10T_SystemGain_20160225
    # a = TipIn(file_path='./function_stlye_module/tip in/20180314_171727(362_6)_tip_in_E.csv', teststr='test')
    a = TipIn(file_path='D:/pythonCodes/function_stlye_module/tip in/20180314_171727(362_6)_tip_in_E.csv', teststr='test')
    a.tipin_main()
    # plt.show()

    print('Finish!')
