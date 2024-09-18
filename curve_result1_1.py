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

    legend_entries = []
    plt.figure(figsize=(16, 7))
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
            res1_1.append(result[0][2])
            res2_1.append(result[1][2])
            path = './data/results/inference_final_results/test_results_' + datasets[
                i] + '_preactresnet18_rdp_' + e + '.csv'
            result = np.array(pd.read_csv(path, header=None, low_memory=False))
            res1_2.append(result[0][2])
            res2_2.append(result[1][2])
            path = './data/results/inference_final_results/test_results_' + datasets[i] + '_vgg19_rdp_' + e + '.csv'
            result = np.array(pd.read_csv(path, header=None, low_memory=False))
            res1_3.append(result[0][2])
            res2_3.append(result[1][2])
        path = './data/results/inference_final_results/test_results_' + datasets[i] + '_cnn.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_1.append(result[0][2])
        res2_1.append(result[1][2])
        path = './data/results/inference_final_results/test_results_' + datasets[i] + '_preactresnet18.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_2.append(result[0][2])
        res2_2.append(result[1][2])
        path = './data/results/inference_final_results/test_results_' + datasets[i] + '_vgg19.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_3.append(result[0][2])
        res2_3.append(result[1][2])
        bar_width = 0.2  # 条形宽度
        a = np.arange(5)  # bar1的横坐标
        b = a + bar_width + 0.02  # bar2横坐标
        c = b + bar_width + 0.02  # bar2横坐标

        plt.subplot(2, 4, i + 1)
        # plt.figure(figsize=(8, 6))
        CNN = plt.bar(a, height=res1_1, width=bar_width, color='#6C7081', label='SimpleCNN')
        legend_entries.append(CNN)
        ResNet18 = plt.bar(b, height=res1_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
        ResNet18 = legend_entries.append(ResNet18)
        VGG19 = plt.bar(c, height=res1_3, width=bar_width, color='#638a73', label='VGG19')
        legend_entries.append(VGG19)
        # plt.legend(fontsize=11)  # 显示图例
        x = ("$\epsilon$=2", "$\epsilon$=4", "$\epsilon$=6", "$\epsilon$=8", 'None')
        plt.xticks(a + bar_width + 0.01, x, fontsize=14)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
        # plt.xticks([])  # 设置x轴刻度的显示位置s， a + bar_width/2 为横坐标轴刻度的位置
        plt.yticks(fontsize=14)
        # plt.xlabel('Privacy Budget')  # 纵坐标轴标题
        if i == 0:
            plt.ylabel('Accuracy', fontsize=15)  # 纵坐标轴标题
        plt.ylim(0, 1)
        plt.title(names[i], fontsize=15)  # 图形标题
        plt.grid(linestyle='--', linewidth=0.7, axis='y')

        plt.subplot(2, 4, i + 5)
        # plt.figure(figsize=(8, 6))
        CNN = plt.bar(a, height=res2_1, width=bar_width, color='#6C7081', label='SimpleCNN')
        legend_entries.append(CNN)
        plt.bar(b, height=res2_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
        ResNet18 = legend_entries.append(ResNet18)
        VGG19 = plt.bar(c, height=res2_3, width=bar_width, color='#638a73', label='VGG19')
        legend_entries.append(VGG19)
        # plt.legend(fontsize=10)  # 显示图例
        x = ("$\epsilon$=2", "$\epsilon$=4", "$\epsilon$=6", "$\epsilon$=8", 'None')
        plt.xticks(a + bar_width + 0.01, x, fontsize=14)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
        plt.yticks(fontsize=14)
        plt.xlabel('Privacy Budget', fontsize=15)  # 纵坐标轴标题
        if i == 0:
            plt.ylabel('Accuracy', fontsize=15)  # 纵坐标轴标题
        plt.ylim(0, 1)
        # plt.title(names[i])  # 图形标题
        plt.grid(linestyle='--', linewidth=0.7, axis='y')
    plt.legend(legend_entries, labels=['SimpleCNN', 'PreAct-ResNet18', 'VGG19'], loc='lower center', ncol=3,
               fontsize=15)
    plt.savefig("./data/figure1.pdf")
    plt.show()

    legend_entries = []
    plt.figure(figsize=(16, 7))
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

        plt.subplot(2, 4, i + 1)
        # plt.figure(figsize=(8, 6))
        CNN = plt.bar(a, height=res1_1, width=bar_width, color='#6C7081', label='SimpleCNN')
        legend_entries.append(CNN)
        ResNet18 = plt.bar(b, height=res1_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
        ResNet18 = legend_entries.append(ResNet18)
        VGG19 = plt.bar(c, height=res1_3, width=bar_width, color='#638a73', label='VGG19')
        legend_entries.append(VGG19)
        # plt.legend(fontsize=11)  # 显示图例
        x = ("$\epsilon$=2", "$\epsilon$=4", "$\epsilon$=6", "$\epsilon$=8", 'None')
        plt.xticks(a + bar_width + 0.01, x, fontsize=14)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
        # plt.xticks([])  # 设置x轴刻度的显示位置s， a + bar_width/2 为横坐标轴刻度的位置
        plt.yticks(fontsize=14)
        # plt.xlabel('Privacy Budget')  # 纵坐标轴标题
        if i == 0:
            plt.ylabel('MSE', fontsize=15)  # 纵坐标轴标题
        plt.ylim(0, 1)
        plt.title(names[i], fontsize=15)  # 图形标题
        plt.grid(linestyle='--', linewidth=0.7, axis='y')

        plt.subplot(2, 4, i + 5)
        # plt.figure(figsize=(8, 6))
        CNN = plt.bar(a, height=res2_1, width=bar_width, color='#6C7081', label='SimpleCNN')
        legend_entries.append(CNN)
        plt.bar(b, height=res2_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
        ResNet18 = legend_entries.append(ResNet18)
        VGG19 = plt.bar(c, height=res2_3, width=bar_width, color='#638a73', label='VGG19')
        legend_entries.append(VGG19)
        # plt.legend(fontsize=10)  # 显示图例
        x = ("$\epsilon$=2", "$\epsilon$=4", "$\epsilon$=6", "$\epsilon$=8", 'None')
        plt.xticks(a + bar_width + 0.01, x, fontsize=14)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
        plt.yticks(fontsize=14)
        plt.xlabel('Privacy Budget', fontsize=15)  # 纵坐标轴标题
        if i == 0:
            plt.ylabel('Accuracy', fontsize=15)  # 纵坐标轴标题
        plt.ylim(0, 1)
        # plt.title(names[i])  # 图形标题
        plt.grid(linestyle='--', linewidth=0.7, axis='y')
    plt.legend(legend_entries, labels=['SimpleCNN', 'PreAct-ResNet18', 'VGG19'], loc='lower center', ncol=3,
               fontsize=15)
    plt.savefig("./data/figure2.pdf")
    plt.show()



    plt.figure(figsize=(16, 7))
    res1_1 = []
    res1_2 = []
    res1_3 = []
    for e in epsilon:
        path = './data/results/inference_final_results/test_results_utkface_cnn_rdp_' + e + '.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_1.append(result[4][2])
        path = './data/results/inference_final_results/test_results_utkface_preactresnet18_rdp_' + e + '.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_2.append(result[4][2])
        path = './data/results/inference_final_results/test_results_utkface_vgg19_rdp_' + e + '.csv'
        result = np.array(pd.read_csv(path, header=None, low_memory=False))
        res1_3.append(result[4][2])
    path = './data/results/inference_final_results/test_results_utkface_cnn.csv'
    result = np.array(pd.read_csv(path, header=None, low_memory=False))
    res1_1.append(result[4][2])
    path = './data/results/inference_final_results/test_results_utkface_preactresnet18.csv'
    result = np.array(pd.read_csv(path, header=None, low_memory=False))
    res1_2.append(result[4][2])
    path = './data/results/inference_final_results/test_results_utkface_vgg19.csv'
    result = np.array(pd.read_csv(path, header=None, low_memory=False))
    res1_3.append(result[4][2])
    bar_width = 0.2  # 条形宽度
    a = np.arange(5)  # bar1的横坐标
    b = a + bar_width + 0.02  # bar2横坐标
    c = b + bar_width + 0.02  # bar2横坐标
    plt.subplot(2, 4, 1)
    plt.bar(a, height=res1_1, width=bar_width, color='#6C7081', label='SimpleCNN')
    plt.bar(b, height=res1_2, width=bar_width, color='#AF855A', label='PreAct-ResNet18')
    plt.bar(c, height=res1_3, width=bar_width, color='#638a73', label='VGG19')
    # plt.legend(fontsize=22, ncol=3)  # 显示图例
    x = ("$\epsilon$=2", "$\epsilon$=4", "$\epsilon$=6", "$\epsilon$=8", 'None')
    plt.xticks(a + bar_width + 0.01, x, fontsize=12)  # 设置x轴刻度的显示位置， a + bar_width/2 为横坐标轴刻度的位置
    plt.yticks(fontsize=12)
    plt.xlabel('Privacy Budget', fontsize=13)  # 纵坐标轴标题
    plt.ylabel('Accuracy', fontsize=13)  # 纵坐标轴标题
    plt.ylim(0, 1)
    plt.title('UTKFace', fontsize=13)  # 图形标题

    plt.legend(legend_entries, labels=['SimpleCNN', 'PreAct-ResNet18', 'VGG19'], ncol=3,
               fontsize=13)
    # plt.savefig("./data/figure3.pdf")
    plt.show()
    # plt.grid(linestyle='--', linewidth=0.7, axis='y')
    # plt.savefig("./data/figure3.pdf")
    # plt.show()
