#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt
import pywt

######### write to excel #########
class PRND(object):
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
        self.prnd_class = {}

    def prnd_main(self):
        '''
        Main function of calculating system gain, save  'self.sysGain_class'  to be called from UI.

        '''
        # parameters setting
        fre = 20  # sample rate
        duration_min = 2  # minimum time for gear holding, used for data filtering
        acc_pos = ['wheel', 'rail']  # can be select within this range
        acc_dir = ['x', 'y', 'z', 'add']
        # parameters setting ends

        # raw data input

        # signal matching
        # acc_data_type = [acc_pos[i] + '_' + acc_dir[j] for i in range(0, len(acc_pos)) for j in range(0, len(acc_dir)-1)]  # this should be written to the pull menu's title
        # acc_data_signal_name = ['Acc_Wx', 'Acc_Wy', 'Acc_Wz', 'Acc_Rx', 'Acc_Ry', 'Acc_Rz']  # this should be return from the pull menu selection, with default value
        # acc_dict = {acc_data_type[i]: acc_data_signal_name[i] for i in range(0, len(acc_data_type))}

        prnd_data_ful = pd.read_csv(self.filepath)
        prnd_data_selc = prnd_data_ful.loc[:, ['Acc_Rx', 'Acc_Wx', 'Acc_Ry', 'Acc_Wy', 'Acc_Rz', 'Acc_Wz', 'Gear']]

        # implicit data input
        # for i in range(0, len(acc_data_type)):
        #     exec('acc_' + acc_data_type[i] + '_data = prnd_data_selc[acc_dict[\'' + acc_data_type[i] + '\']].tolist()')

        # explicit data input
        acc_rail_x_data = prnd_data_selc['Acc_Rx'].tolist()
        acc_wheel_x_data = prnd_data_selc['Acc_Wx'].tolist()
        acc_rail_y_data = prnd_data_selc['Acc_Ry'].tolist()
        acc_wheel_y_data = prnd_data_selc['Acc_Wy'].tolist()
        acc_rail_z_data = prnd_data_selc['Acc_Rz'].tolist()
        acc_wheel_z_data = prnd_data_selc['Acc_Wz'].tolist()
        gear_data = prnd_data_selc['Gear'].tolist()
        abstime_data = [i/fre for i in range(0, len(gear_data))]
        # raw data input ends

        # gear signal matching & transfer: will be done with a pull menu
        gear_type = np.unique(gear_data).tolist()  # each element in gear_type will be written to the pull menu corresponding to PRND for user selecting
        name_p = 'P'  # this should be returned by the pull menu
        name_r = 'R'
        name_n = 'N'
        name_d = 'D'
        gear_dict = {str(name_p): 1, str(name_r): 2, str(name_n): 3, str(name_d): 4}  # PRND are presented by 1234 respectively
        gear_digit = [gear_dict[x] if x in gear_dict else 0 for x in gear_data]  # gear signal transfer to digit
        # gear signal matching & transfer ends

        # data segment & filtering: only segment shifting within PRND and with duration larger than 2s
        gear_tran_index = [i for i in range(0, len(gear_digit)-1) if gear_digit[i+1] != gear_digit[i]]
        start_index = [i+1 for i in gear_tran_index if (gear_digit[i+1]*gear_digit[i] != 0)]
        end_index = [i for i in gear_tran_index if (gear_digit[i+1]*gear_digit[i] != 0)]
        if end_index[0] != gear_tran_index[0]:  # if the first segment is not valid due to fault gear position, then use the index of previous gear transion as start of the first segment
            start_index.insert(0, gear_tran_index[gear_tran_index.index(end_index[0])-1]+1)
        else:
            start_index.insert(0,0)
        if end_index[-1] != gear_tran_index[-1]:
            end_index.append(gear_tran_index[gear_tran_index.index(end_index[-1])+1])
        else:
            end_index.append(len(gear_digit))

        start_index_filtered = [start_index[i] for i in range(0, len(start_index)) if (end_index[i]-start_index[i]) >= fre*duration_min]
        end_index_filtered = [end_index[i] for i in range(0, len(end_index)) if (end_index[i]-start_index[i]) >= fre*duration_min]
        gear_sequence = [gear_digit[i] for i in start_index_filtered]
        valid_tran_flag = [1 if (start_index_filtered[i+1]-end_index_filtered[i] == 1) else 0 for i in range(0, len(start_index_filtered)-1)]  # adjacent shifter position must be continuous in time so that it can be consider as valid shift
        # data segment & filtering ends

        # # accerlation signal extract for unique shifter position
        # acc_rail_x_hold = [acc_rail_x_data[start_index_filtered[i]: end_index_filtered[i]] for i in range(0, len(start_index_filtered))]
        # acc_rail_y_hold = [acc_rail_y_data[start_index_filtered[i]: end_index_filtered[i]] for i in range(0, len(start_index_filtered))]
        # acc_rail_z_hold = [acc_rail_z_data[start_index_filtered[i]: end_index_filtered[i]] for i in range(0, len(start_index_filtered))]
        # acc_wheel_x_hold = [acc_wheel_x_data[start_index_filtered[i]: end_index_filtered[i]] for i in range(0, len(start_index_filtered))]
        # acc_wheel_y_hold = [acc_wheel_y_data[start_index_filtered[i]: end_index_filtered[i]] for i in range(0, len(start_index_filtered))]
        # acc_wheel_z_hold = [acc_wheel_z_data[start_index_filtered[i]: end_index_filtered[i]] for i in range(0, len(start_index_filtered))]
        # acc_rail_add_hold = [[(acc_rail_x_hold[i][j]**2 + acc_rail_y_hold[i][j]**2 + acc_rail_z_hold[i][j]**2)**0.5 for j in range(0, len(acc_rail_x_hold[i]))] for i in range(0, len(acc_rail_x_hold))]
        # acc_wheel_add_hold = [[(acc_wheel_x_hold[i][j]**2 + acc_wheel_y_hold[i][j]**2 + acc_wheel_z_hold[i][j]**2)**0.5 for j in range(0, len(acc_wheel_x_hold[i]))] for i in range(0, len(acc_wheel_x_hold))]
        #
        # # result packing
        # acc_segment = {}
        # acc_segment['rail'] = {'x': acc_rail_x_hold, 'y': acc_rail_y_hold, 'z': acc_rail_z_hold, 'add': acc_rail_add_hold}
        # acc_segment['wheel'] = {'x': acc_wheel_x_hold, 'y': acc_wheel_y_hold, 'z': acc_wheel_z_hold, 'add': acc_wheel_add_hold}
        # # accerlation signal extract for unique shifter position ends


        # accerlation signal extract for two subsequent shifter position and calculation of RMS
        # execution error: implicit referene to variable fail during execution but fine while executed in console
        # for i in range(0, len(acc_pos)):
        #     for j in range(0, len(acc_dir)):
        #         exec('acc_' + acc_pos[i] + '_' + acc_dir[j] + '_extract = [acc_' + acc_pos[i] + '_' + acc_dir[j] + '_data[end_index_filtered[t] - fre*1: end_index_filtered[t] + 1 + fre*1] for t in range(0, len(end_index_filtered)-1)]')
        end_index_filtered_valid_trans = [end_index_filtered[i] for i in range(0, len(end_index_filtered)-1) if valid_tran_flag[i] == 1]
        acc_rail_x_trans = [acc_rail_x_data[end_index_filtered_valid_trans[i] - fre*1: end_index_filtered_valid_trans[i] + 1 + fre*1] for i in range(0, len(end_index_filtered_valid_trans))]
        acc_wheel_x_trans = [acc_wheel_x_data[end_index_filtered_valid_trans[i] - fre*1: end_index_filtered_valid_trans[i] + 1 + fre*1] for i in range(0, len(end_index_filtered_valid_trans))]
        acc_rail_y_trans = [acc_rail_y_data[end_index_filtered_valid_trans[i] - fre*1: end_index_filtered_valid_trans[i] + 1 + fre*1] for i in range(0, len(end_index_filtered_valid_trans))]
        acc_wheel_y_trans = [acc_wheel_y_data[end_index_filtered_valid_trans[i] - fre*1: end_index_filtered_valid_trans[i] + 1 + fre*1] for i in range(0, len(end_index_filtered_valid_trans))]
        acc_rail_z_trans = [acc_rail_z_data[end_index_filtered_valid_trans[i] - fre*1: end_index_filtered_valid_trans[i] + 1 + fre*1] for i in range(0, len(end_index_filtered_valid_trans))]
        acc_wheel_z_trans = [acc_wheel_z_data[end_index_filtered_valid_trans[i] - fre*1: end_index_filtered_valid_trans[i] + 1 + fre*1] for i in range(0, len(end_index_filtered_valid_trans))]
        acc_rail_add_trans = [[(acc_rail_x_trans[i][j]**2 + acc_rail_y_trans[i][j]**2 + acc_rail_z_trans[i][j]**2)**0.5 for j in range(0, len(acc_rail_x_trans[i]))] for i in range(0, len(acc_rail_x_trans))]
        acc_wheel_add_trans = [[(acc_wheel_x_trans[i][j]**2 + acc_wheel_y_trans[i][j]**2 + acc_wheel_z_trans[i][j]**2)**0.5 for j in range(0, len(acc_wheel_x_trans[i]))] for i in range(0, len(acc_wheel_x_trans))]

        # index caculation--------------------------------------------------------------
        acc_rms_rail_x = [PRND.cal_rms(acc_rail_x_trans[i]) for i in range(0, len(acc_rail_x_trans))]
        acc_rms_rail_y = [PRND.cal_rms(acc_rail_y_trans[i]) for i in range(0, len(acc_rail_y_trans))]
        acc_rms_rail_z = [PRND.cal_rms(acc_rail_z_trans[i]) for i in range(0, len(acc_rail_z_trans))]
        acc_rms_rail_add = [PRND.cal_rms(acc_rail_add_trans[i]) for i in range(0, len(acc_rail_add_trans))]
        acc_rms_wheel_x = [PRND.cal_rms(acc_wheel_x_trans[i]) for i in range(0, len(acc_wheel_x_trans))]
        acc_rms_wheel_y = [PRND.cal_rms(acc_wheel_y_trans[i]) for i in range(0, len(acc_wheel_y_trans))]
        acc_rms_wheel_z = [PRND.cal_rms(acc_wheel_z_trans[i]) for i in range(0, len(acc_wheel_z_trans))]
        acc_rms_wheel_add = [PRND.cal_rms(acc_wheel_add_trans[i]) for i in range(0, len(acc_wheel_add_trans))]

        # rms result packing
        acc_rms = {}
        acc_rms['rail'] = {'x': acc_rms_rail_x, 'y': acc_rms_rail_y, 'z': acc_rms_rail_z, 'add': acc_rms_rail_add}
        acc_rms['wheel'] = {'x': acc_rms_wheel_x, 'y': acc_rms_wheel_y, 'z': acc_rms_wheel_z, 'add': acc_rms_wheel_add}
        # end of index calculation---------------------------------------------------------

        # segment data packing
        acc_segment = {}
        acc_segment['rail'] = {'x': acc_rail_x_trans, 'y': acc_rail_y_trans, 'z': acc_rail_z_trans, 'add': acc_rail_add_trans}
        acc_segment['wheel'] = {'x': acc_wheel_x_trans, 'y': acc_wheel_y_trans, 'z': acc_wheel_z_trans, 'add': acc_wheel_add_trans}
        # accerlation signal extract for two subsequent shifter position and calculation of RMS ends


        PRND.plot_curve(gear_sequence, valid_tran_flag, fre, acc_rms, acc_segment)

    @ staticmethod
    def cal_rms(acc_data):
        acc_rms = (sum([acc_data[i]**2 for i in range(0, len(acc_data))])/len(acc_data))**0.5
        return acc_rms

    @ staticmethod
    def plot_curve(gear_sequence, valid_flag, fre, acc_rms, acc_segment):
        fig1 = plt.figure()

        # colormap
        colormap = []
        colormap.append([0.8431, 0.6781, 0.2941])
        colormap.append([0.9061, 0.5491, 0.2241])
        colormap.append([0.5181, 0.2631, 0.1531])
        colormap.append([0.3101, 0.3181, 0.1651])
        color_dict = {'x': 0, 'y': 1, 'z': 2, 'add': 3}

        pos_dict = {'rail': 1, 'wheel': 2}
        legend_dict = {'x': 'x direction', 'y': 'y direction', 'z': 'z direction', 'add': 'add'}
        legend_dis = {'rail': [], 'wheel': []}

        gear_dict = {1: 'P', 2: 'R', 3: 'N', 4: 'D'}
        gear_sequence_label = [gear_dict[gear_sequence[i]] for i in range(0, len(gear_sequence))]  # digital gear 1234 trans back to label PRND
        tran_label = [gear_sequence_label[i] + '->' + gear_sequence_label[i+1] for i in range(0, len(gear_sequence)-1) if valid_flag[i] == 1]

        for key_pos in acc_rms:
            for key_dir in acc_rms[key_pos]:
                # color & position determination
                color_index = color_dict[key_dir]
                legend_dis[key_pos].append(legend_dict[key_dir])
                subfigure_index = pos_dict[key_pos]

                # rms curves
                ax1 = fig1.add_subplot(1, 2, subfigure_index)
                ax1.plot(range(len(tran_label)), acc_rms[key_pos][key_dir],  'ro', color=colormap[color_index])
                plt.xticks(range(len(tran_label)), tran_label)
                ax1.set_xlabel('Shifter status ', fontsize=10)
                ax1.set_ylabel('RMS of Acc (g)', fontsize=10)
                ax1.set_title(key_pos, fontsize=12)
                ax1.legend(legend_dis[key_pos], loc=2, fontsize=15)
                ax1.set_ylim(0, 0.15)
        # plt.show()

        legend_dis = [[]*i for i in range(0, 2 * sum(valid_flag))]
        fig2 = plt.figure()
        time = [i/fre for i in range(-fre, fre+1)]
        for key_pos in acc_segment:
            for key_dir in acc_segment[key_pos]:
                # color & position determination
                color_index = color_dict[key_dir]
                for i in range(0, len(acc_segment[key_pos][key_dir])):
                    subfigure_index = (pos_dict[key_pos]-1)*sum(valid_flag) + i + 1
                    legend_dis[subfigure_index-1].append(legend_dict[key_dir])
                    ax1 = fig2.add_subplot(2, sum(valid_flag), subfigure_index)
                    ax1.plot(time, acc_segment[key_pos][key_dir][i], linewidth=2, color=colormap[color_index])
                    if i == 0:
                        ax1.set_xlabel('Time (s) ', fontsize=10)
                        ax1.set_ylabel('Acc (g)', fontsize=10)
                    ax1.set_title(tran_label[i] + ':' + key_pos, fontsize=12)
                    ax1.legend(legend_dis[i], loc=2, fontsize=15)
                    ax1.set_ylim(0, 0.2)
        plt.show()



if __name__ == '__main__':
    # *******1-GetSysGainData****** AS22_C16UVV016_SystemGain_20160925_D_M_SL, IP31_L16UOV055_10T_SystemGain_20160225
    # a = TipIn(file_path='./function_stlye_module/tip in/20180314_171727(362_6)_tip_in_E.csv', teststr='test')
    a = PRND(file_path='D:/pythonCodes/function_stlye_module/PRND/PRND.csv', teststr='test')
    a.prnd_main()
    # plt.show()

    print('Finish!')
