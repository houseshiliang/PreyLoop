from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns

plt.rc('font', family='Times New Roman')
# plt.rc('font', family='Arial')
np.random.seed(11)

if __name__ == "__main__":
    datasets = ['fmnist', 'utkface', 'stl10', 'cifar10']
    # datasets = ['fmnist']
    names = ['FMNIST', 'UTKFace', 'STL10', 'CIFAR10']
    models = ['cnn', 'preactresnet18', 'vgg19']
    epsilon = ['2', '4', '6', '8']


    # for i in range(len(datasets)):
    #     res1_1 = []
    #     res1_2 = []
    #     res1_3 = []
    #     res2_1 = []
    #     res2_2 = []
    #     res2_3 = []
    #     for e in epsilon:
    #         path = './data/results/inference_final_results/test_results_' + datasets[i] + '_cnn_rdp_' + e + '.csv'
    #         result = np.array(pd.read_csv(path, header=None, low_memory=False))
    #         res1_1.append(result[0][2])
    #         res2_1.append(result[1][2])
    #         path = './data/results/inference_final_results/test_results_' + datasets[
    #             i] + '_preactresnet18_rdp_' + e + '.csv'
    #         result = np.array(pd.read_csv(path, header=None, low_memory=False))
    #         res1_2.append(result[0][2])
    #         res2_2.append(result[1][2])
    #         path = './data/results/inference_final_results/test_results_' + datasets[i] + '_vgg19_rdp_' + e + '.csv'
    #         result = np.array(pd.read_csv(path, header=None, low_memory=False))
    #         res1_3.append(result[0][2])
    #         res2_3.append(result[1][2])
    #     path = './data/results/inference_final_results/test_results_' + datasets[i] + '_cnn.csv'
    #     result = np.array(pd.read_csv(path, header=None, low_memory=False))
    #     res1_1.append(result[0][2])
    #     res2_1.append(result[1][2])
    #     path = './data/results/inference_final_results/test_results_' + datasets[i] + '_preactresnet18.csv'
    #     result = np.array(pd.read_csv(path, header=None, low_memory=False))
    #     res1_2.append(result[0][2])
    #     res2_2.append(result[1][2])
    #     path = './data/results/inference_final_results/test_results_' + datasets[i] + '_vgg19.csv'
    #     result = np.array(pd.read_csv(path, header=None, low_memory=False))
    #     res1_3.append(result[0][2])
    #     res2_3.append(result[1][2])
    #     print(res1_1)
    #     print(res1_2)
    #     print(res1_3)
    #     print(res2_1)
    #     print(res2_2)
    #     print(res2_3)
    #
    #     bar_width = 0.2  # 条形宽度
    #     a = np.arange(5)  # bar1的横坐标
    #     b = a + bar_width + 0.02  # bar2横坐标
    #     c = b + bar_width + 0.02  # bar2横坐标
    #     # caddd1    4a565f    719F85
    #     # 1B5176    977952    9FADBD
    #     # B6BBBF    4a565f    598E81
    #     # 002fa6 克莱因蓝    470024 勃艮第红  01847f 马歇尔绿  492d22 凡戴克棕  003153 普鲁士蓝
    #     plt.bar(a, height=res1_1, width=bar_width, color='#6C7081', label='SimpleCNN')
    #     plt.bar(b, height=res1_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
    #     plt.bar(c, height=res1_3, width=bar_width, color='#638a73', label='VGG19')
    #     # plt.legend(fontsize=11)  # 显示图例
    #     x = ("$\epsilon=2$", "$\epsilon=4$", "$\epsilon=6$", "$\epsilon=8$", 'None')
    #     plt.xticks(a + bar_width + 0.01, x, fontsize=21)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
    #     plt.yticks(fontsize=21)
    #     plt.xlabel('Privacy Budget', fontsize=22)  # 纵坐标轴标题
    #     plt.ylabel('Accuracy', fontsize=22)  # 纵坐标轴标题
    #     plt.ylim(0, 1)
    #     plt.title(names[i], fontsize=24)  # 图形标题
    #     plt.grid(linestyle='--', linewidth=0.7, axis='y')
    #     plt.savefig("./data/figure1_"+str(i+1)+"_1.pdf")
    #     plt.show()
    #
    #     plt.bar(a, height=res2_1, width=bar_width, color='#6C7081', label='SimpleCNN')
    #     plt.bar(b, height=res2_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
    #     plt.bar(c, height=res2_3, width=bar_width, color='#638a73', label='VGG19')
    #     # plt.legend(fontsize=11)  # 显示图例
    #     x = ("$\epsilon=2$", "$\epsilon=4$", "$\epsilon=6$", "$\epsilon=8$", 'None')
    #     plt.xticks(a + bar_width + 0.01, x, fontsize=21)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
    #     plt.yticks(fontsize=21)
    #     plt.xlabel('Privacy Budget', fontsize=22)  # 纵坐标轴标题
    #     plt.ylabel('Accuracy', fontsize=22)  # 纵坐标轴标题
    #     plt.ylim(0, 1)
    #     plt.title(names[i], fontsize=24)  # 图形标题
    #     plt.grid(linestyle='--', linewidth=0.7, axis='y')
    #     plt.savefig("./data/figure1_"+str(i+1)+"_2.pdf")
    #     plt.show()



    for i in range(len(datasets)):
        res1_1 = []
        res1_2 = []
        res1_3 = []
        res2_1 = []
        res2_2 = []
        res2_3 = []
        for e in epsilon:
            path = './data/results/inference_final_results/test_results_' + datasets[i] + '_cnn_rdp_' + e + '.csv'
            result = np.array(pd.read_csv(path, header=None, low_memory=False))
            res1_1.append(result[2][2])
            res2_1.append(result[3][2])
            path = './data/results/inference_final_results/test_results_' + datasets[
                i] + '_preactresnet18_rdp_' + e + '.csv'
            result = np.array(pd.read_csv(path, header=None, low_memory=False))
            res1_2.append(result[2][2])
            res2_2.append(result[3][2])
            path = './data/results/inference_final_results/test_results_' + datasets[i] + '_vgg19_rdp_' + e + '.csv'
            result = np.array(pd.read_csv(path, header=None, low_memory=False))
            res1_3.append(result[2][2])
            res2_3.append(result[3][2])
        path = './data/results/inference_final_results/test_results_' + datasets[i] + '_cnn.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_1.append(result[2][2])
        res2_1.append(result[3][2])
        path = './data/results/inference_final_results/test_results_' + datasets[i] + '_preactresnet18.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_2.append(result[2][2])
        res2_2.append(result[3][2])
        path = './data/results/inference_final_results/test_results_' + datasets[i] + '_vgg19.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_3.append(result[2][2])
        res2_3.append(result[3][2])
        bar_width = 0.2  # 条形宽度
        a = np.arange(5)  # bar1的横坐标
        b = a + bar_width + 0.02  # bar2横坐标
        c = b + bar_width + 0.02  # bar2横坐标
        plt.bar(a, height=res1_1, width=bar_width, color='#6C7081', label='SimpleCNN')
        plt.bar(b, height=res1_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
        plt.bar(c, height=res1_3, width=bar_width, color='#638a73', label='VGG19')
        # plt.legend(fontsize=11)  # 显示图例
        x = ("$\epsilon=2$", "$\epsilon=4$", "$\epsilon=6$", "$\epsilon=8$", 'None')
        plt.xticks(a + bar_width + 0.01, x, fontsize=21)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
        plt.yticks(fontsize=21)
        plt.xlabel('Privacy Budget', fontsize=22)  # 纵坐标轴标题
        plt.ylabel('MSE', fontsize=22)  # 纵坐标轴标题
        plt.ylim(0, 1)
        plt.title(names[i], fontsize=24)  # 图形标题
        plt.grid(linestyle='--', linewidth=0.7, axis='y')
        # plt.savefig("./data/figure2_"+str(i+1)+"_1.pdf")
        plt.show()

        plt.bar(a, height=res2_1, width=bar_width, color='#6C7081', label='SimpleCNN')
        plt.bar(b, height=res2_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
        plt.bar(c, height=res2_3, width=bar_width, color='#638a73', label='VGG19')
        # plt.legend(fontsize=10)  # 显示图例
        x = ("$\epsilon=2$", "$\epsilon=4$", "$\epsilon=6$", "$\epsilon=8$", 'None')
        plt.xticks(a + bar_width + 0.01, x, fontsize=21)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
        plt.yticks(fontsize=21)
        plt.xlabel('Privacy Budget', fontsize=22)  # 纵坐标轴标题
        plt.ylabel('Accuracy', fontsize=22)  # 纵坐标轴标题
        plt.ylim(0, 1)
        plt.title(names[i], fontsize=24)  # 图形标题
        plt.grid(linestyle='--', linewidth=0.7, axis='y')
        # plt.savefig("./data/figure2_"+str(i+1)+"_2.pdf")
        plt.show()

