#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt


class FixedGearAcc(object):
    """
       Main class of system gain, contains all the thing needed to be calculated or plotted.

       Contains：
       ******
       fun sg_main————Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.
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
        self.fixedgearacc_class = {}

    def fga_main(self):
        '''
        Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.

        '''
        fga_Data_ful = pd.read_csv(self.filepath)
        fga_Data_Selc = fga_Data_ful.loc[:, ['AccPedPos', 'EngineSpeed', 'EnToq', 'GearPos']]

        pedal_data =fga_Data_Selc['AccPedPos'].tolist()
        speed_data = fga_Data_Selc['EngineSpeed'].tolist()
        torque_data = fga_Data_Selc['EnToq'].tolist()
        gear_data = fga_Data_Selc['GearPos'].tolist()
        torque_max = max(torque_data)

        gear_max = int(max(gear_data))
        pedal_range = np.linspace(10, 100, 10)

        # data arrangement
        #            Pedal               Pedal            Pedal
        # -----------------------------------------------------------
        # Gear | np.[EnSpd, Toq]    np.[EnSpd, Toq]   np.[EnSpd, Toq]
        # Gear | np.[EnSpd, Toq]    np.[EnSpd, Toq]   np.[EnSpd, Toq]
        # Gear | np.[EnSpd, Toq]    np.[EnSpd, Toq]   np.[EnSpd, Toq]
        # pedalmap_data = [gear_max, len(pedal_range)]


        # data picking by hold time: fixed pedal and gear, exit till upshift or pedal deviation

        # parameters setting
        fre = 20  # sample rate
        holdtime = 5  # minimum hold time of the pedal
        holdtime_count_min = fre*holdtime
        pedal_min = 0  # minimum pedal
        pedal_var = 0.5  # pedal variation allowed
        pedal_var_exit = 1  # pedal variation for exit

        # parameters setting ends

        holdtime_count = 0
        flag_holdtime = np.zeros(len(pedal_data))

        for i in range(0, len(pedal_data)-1):
            if (pedal_data[i] > pedal_min) & (abs(pedal_data[i+1]-pedal_data[i]) <= pedal_var) & (gear_data[i+1] == gear_data[i]):
                holdtime_count = holdtime_count + 1
            if (abs(pedal_data[i+1] - pedal_data[i]) > pedal_var_exit) | (gear_data[i+1] > gear_data[i]):
                # 源代码写法，应上面条件判定不符合即推出？否则可能两个条件同时不通过，holdtime不加，i在加，导致退出时往前追溯置1的时候会漏掉开始的点
                if holdtime_count > holdtime_count_min:
                    flag_holdtime[i-holdtime_count:i] = 1  # start point included, end point not
                holdtime_count = 0


        # data picking by speed: to account for low gear position where engine speed rise quickly which can't fullfill hold time
        # 低档位加速度大，一方面，车速上升快，另一方面，速比大，单位车速提升对应发动机转速提升大，因此转速升的快，可能不满足hold time判据
        flag_speed = np.zeros(len(pedal_data))
        speedrise_count = 0
        speed_min = 1500
        speed_max = 5000
        pedal_min = 4
        speed_var = -10

        for i in range(0, len(pedal_data)-1):
            if (speedrise_count == 0) & (speed_data[i] >= speed_min) \
                    & ((speed_data[i+1] - speed_data[i]) > 0) & (pedal_data[i] > pedal_min):
                speedrise_count = 1  # begin counting
            if ((abs(pedal_data[i+1] - pedal_data[i]) < pedal_var) | ((speed_data[i+1] - speed_data[i]) > speed_var)) \
                    & (speedrise_count > 0) & (speed_data[i] <= speed_max):
                speedrise_count = speedrise_count + 1  # counting process

            if (speed_data[i] >= speed_max) & (speedrise_count > 0):  # counting exit 1: validated data
                flag_speed[i-speedrise_count:i] = 1
                speedrise_count = 0

            if (speedrise_count > 0) & (pedal_data[i] < pedal_min) & (speed_data[i] < speed_max):
                # counting exit 2: skip if pedal fall before reaching maximum speed
                # 应加一充分条件：转速波动过大，否则局部波动仍不退出
                speedrise_count = 0
        # data picking by speed ends

        flag_holdtime_or_speed = [int(flag_speed[i]) | int(flag_holdtime[i]) for i in range(0, len(pedal_data))]


        # validated data segment cutting by speed: begins with speed > speed_min+100 all/cut?????
        for i in range(0, len(flag_holdtime_or_speed)-1):
            if (flag_holdtime_or_speed[i] == 0) & (flag_holdtime_or_speed[i+1] == 1) & \
                    (speed_data[i+1] > (speed_min + 100)):
                flag_holdtime_or_speed[i+1] = 0

        # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        # sheet = book.add_sheet('test', cell_overwrite_ok=True)
        # for i in range(0, len(flag_holdtime_or_speed)):
        #     sheet.write(i, 0, flag_holdtime_or_speed[i])
        #     sheet.write(i, 1, flag_holdtime[i])
        # book.save(r'D:\pythonCodes\function_stlye_module\test.xls')

        # validated data segment cutting by speed ends

        # data arrangement by gear position and pedal
        torcurve_start = 0
        torcurve_end = 0
        pedalmap = {}
        for i in range(0, len(flag_holdtime_or_speed)-1):
            if flag_holdtime_or_speed[i] == 0 and flag_holdtime_or_speed[i+1] == 1:
                torcurve_start = i
            if flag_holdtime_or_speed[i] == 1 and flag_holdtime_or_speed[i+1] == 0:
                torcurve_end = i
                torcurve = np.array([speed_data[torcurve_start+1:torcurve_end+1],
                                     torque_data[torcurve_start+1:torcurve_end+1]])
                gear_ave = np.mean(gear_data[torcurve_start+1:torcurve_end])
                pedal_ave = np.mean(pedal_data[torcurve_start + 1:torcurve_end])

                for gear in range(1, gear_max+1):
                    for pedal in pedal_range:
                        if (abs(gear_ave-gear) < 0.5) & (abs(pedal_ave-pedal) < 1):
                            if str(int(gear))+'#'+str(int(pedal))+'%' in pedalmap.keys():
                                if pedalmap[str(int(gear))+'#'+str(int(pedal))+'%'].size < torcurve.size:
                                    pedalmap[str(int(gear))+'#'+str(int(pedal))+'%'] = torcurve
                            else:
                                pedalmap[str(int(gear))+'#'+str(int(pedal))+'%'] = torcurve

                torcurve_start = 0
                torcurve_end = 0

        # data arrangement ends

        FixedGearAcc.plot_torcurve(self, pedalmap, gear_max, pedal_range, fre, torque_max)

    def plot_torcurve(self, pedalmap, gear_max, pedal_range, fre, torque_max):
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()

        color_gear = []
        color_gear.append([i/255 for i in [255, 78, 2]])
        color_gear.append([i/255 for i in [255, 178, 0]])
        color_gear.append([i/255 for i in [255, 217, 0]])
        color_gear.append([i/255 for i in [129, 209, 20]])
        color_gear.append([i/255 for i in [10, 159, 152]])
        color_gear.append([i/255 for i in [0, 81, 161]])
        color_gear.append([i/255 for i in [151, 1, 126]])

        for key in pedalmap:
            gear = key[0]
            pedal = key[2:-1]
            curve = pedalmap[key]
            speed = curve[0]
            torque = curve[1]
            
            color_index = eval(pedal)/100
            if color_index < 0.5:
                color_r = 0.9961 * (0.5 - color_index) * 2 + color_index * 2 * 0.8941
                color_g = 0.9221 * (0.5 - color_index) * 2 + color_index * 2 * 0.3221
                color_b = 0.3961 * (0.5 - color_index) * 2 + color_index * 2 * 0.1061
            else:
                color_r = 0.8941 * (1 - color_index) * 2 + (color_index - 0.5) * 2 * 0.3021
                color_g = 0.3221 * (1 - color_index) * 2 + (color_index - 0.5) * 2 * 0.2041
                color_b = 0.1061 * (1 - color_index) * 2 + (color_index - 0.5) * 2 * 0.1841
                
            self.fig1.add_subplot(2, 4, eval(gear)).plot(speed[int(fre/4):-fre+1], torque[int(fre/4):-fre+1],
                                                         color=[color_r, color_g, color_b])
            self.fig1.add_subplot(2, 4, eval(gear)).set_xlabel('Speed (rpm)', fontsize=12)
            self.fig1.add_subplot(2, 4, eval(gear)).set_ylabel('Torque (Nm)', fontsize=12)
            self.fig1.add_subplot(2, 4, eval(gear)).set_title(gear, fontsize=12)
            self.fig1.add_subplot(2, 4, eval(gear)).set_xlim(1000, 6500)
            self.fig1.add_subplot(2, 4, eval(gear)).set_ylim(0, torque_max+20)

            self.fig2.add_axes([0.15, 0.1, 0.75, 0.8]).plot(speed[int(fre/4):-fre+1], torque[int(fre/4):-fre+1],
                           color=color_gear[eval(gear)-1], linewidth=int(eval(pedal)/50+1))
            self.fig2.add_axes([0.15, 0.1, 0.75, 0.8]).set_xlabel('Speed (rpm)', fontsize=12)
            self.fig2.add_axes([0.15, 0.1, 0.75, 0.8]).set_ylabel('Torque (Nm)', fontsize=12)
            self.fig2.add_axes([0.15, 0.1, 0.75, 0.8]).set_title('Pedal map by gear color', fontsize=12)
            self.fig2.add_axes([0.15, 0.1, 0.75, 0.8]).set_xlim(1000, 6500)
            self.fig2.add_axes([0.15, 0.1, 0.75, 0.8]).set_ylim(0, torque_max+20)

        plt.show()








if __name__ == '__main__':
    # *******1-GetSysGainData****** AS22_C16UVV016_SystemGain_20160925_D_M_SL, IP31_L16UOV055_10T_SystemGain_20160225
    a = FixedGearAcc(file_path='./FixGear_Comfort.csv', teststr='test')
    a.fga_main()
    # plt.show()

    print('Finish!')
