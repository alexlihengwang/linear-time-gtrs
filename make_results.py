import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


date = "20230109/"

path_outputs = "outputs_" + date   # script ouputs
path_results = "results_" + date   # dir to save figures and tables
if not os.path.exists(path_results):
    os.mkdir(path_results)


#########################
# make figures 1--3

n_list = ['1000', '10000', '100000']
allData = {}

for n in n_list:
    if n == '1000':
        path = path_outputs + "results_server_1e3"
        algs = ['JL19', 'BTH14', 'AN19', 'WK20', 'WLK21']
        # algs = ['WLK21(ErrorCR)', 'WLK21']
        numTests = 100

    elif n == '10000':
        path = path_outputs + "results_server_1e4"
        algs = ['JL19', 'AN19', 'WK20', 'WLK21']
        # algs = ['WLK21(ErrorCR)', 'WLK21']
        numTests = 100
    else:
        path = path_outputs + "results_server_1e5"
        algs = ['JL19', 'WK20', 'WLK21']
        # algs = ['WLK21(ErrorCR)', 'WLK21']
        numTests = 5

    data = []
    
    for N in [100, 10]:
        for mu in ['1e-06', '1e-04', '1e-02']:
            graphData = []
            for i in range(len(algs)):
                graphData.append([])
                reader = csv.reader(open(path + "/iterates_" + algs[i] + "_N=" + str(N) + "_mu=" + mu + "_n=" + n + ".csv"), delimiter=",")
                for _ in range(numTests):
                    x = next(reader)
                    y = next(reader)
                    graphData[i].append([float(x_) for x_ in x])
                    graphData[i].append([float(y_) for y_ in y])

            data.append({
                'mu': mu,
                'N': N,
                'graph': graphData
                })

    allData[n] = data

plt.rcParams.update({'font.size': 14})
plt.rc('text', usetex=True)

for n in n_list:
    if n == '1000':
        numTests = 100
        algs = ['JL19', 'BTH14', 'AN19', 'WK20', 'WLK21']
        colors = ['#4DAF4A', '#FF7F00', '#E41A1C', '#377EB8', '#984EA3']
        xlims = [30, 20, 10, 3, 2, 1]
        lw, a = 1, 0.25
        filename = '1e3fig.pdf'
    elif n == '10000':
        numTests = 100
        algs = ['JL19',  'AN19', 'WK20', 'WLK21']
        colors = ['#4DAF4A', '#E41A1C', '#377EB8', '#984EA3']
        xlims = [600, 400, 200, 60, 40, 20]
        lw, a = 1, 0.25
        filename = '1e4fig.pdf'
    elif n == '100000':
        numTests = 5
        algs = ['JL19', 'WK20', 'WLK21']
        colors = ['#4DAF4A', '#377EB8', '#984EA3']
        xlims = [9000, 6000, 3000, 900, 600, 300]
        lw, a = 1, 1
        filename = '1e5fig.pdf'

    numAlgs = len(algs)
    fig, axs = plt.subplots(2, 3)

    data = allData[n]

    for graph in range(6):
        graphData = data[graph]['graph']
        title = r'$\bar\mu^* = ' 
        if data[graph]['mu'] == '1e-02':
            title += '10^{-2}'
        elif data[graph]['mu'] == '1e-04':
            title += '10^{-4}'
        else:
            title += '10^{-6}'
        if data[graph]['N'] == 10:
            title += r',\,\bar N = 10^{' + str(len(n)) + r'}$'
        else:
            title += r',\,\bar N = 10^{' + str(len(n) + 1) + r'}$'

        for algNum, algData in enumerate(graphData):
            for i in range(numTests):
                x = np.array(algData[2 * i])
                y = np.array(algData[2 * i + 1])
                x = np.extract(y > 0, x)
                y = np.extract(y > 0, y)
                # print(all(y > 0))
                axs[1 - graph // 3, 2 - graph % 3].plot(x, y, color=colors[algNum], linewidth=lw, alpha=a)

        axs[1 - graph // 3, 2 - graph % 3].set_yscale('log')
        axs[1 - graph // 3, 2 - graph % 3].set_title(title)
        axs[1 - graph // 3, 2 - graph % 3].grid()
        axs[1 - graph // 3, 2 - graph % 3].set_xlim([0,xlims[graph]])
        axs[1 - graph // 3, 2 - graph % 3].set_ylim([1e-14, 1e0])
        axs[1 - graph // 3, 2 - graph % 3].set_xlabel('time (s)')
        axs[1 - graph // 3, 2 - graph % 3].set_ylabel('error')

    patches = []
    for algNum, algName in enumerate(algs):
        patches.append(mpatches.Patch(color=colors[algNum], label=algName))
    patches.reverse()

    if n == '1000':
        patches[3], patches[4] = patches[4], patches[3]

    fig.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, right=1-0.04, top=1-0.075, wspace=0.35, hspace=0.45)
    fig.legend(handles=patches, loc="lower center", ncol=numAlgs)
    fig.set_size_inches(14,7)
    plt.savefig(path_results + filename)


#########################
# make figures 4--6

n_list = ['1000', '10000', '100000']
allData = {}

for n in n_list:
    if n == '1000':
        path = path_outputs + "results_server_1e3"
        algs = ['WLK21(ErrorCR)', 'WLK21']
        numTests = 100

    elif n == '10000':
        path = path_outputs + "results_server_1e4"
        algs = ['WLK21(ErrorCR)', 'WLK21']
        numTests = 100
    else:
        path = path_outputs + "results_server_1e5"
        algs = ['WLK21(ErrorCR)', 'WLK21']
        numTests = 5

    data = []
    
    for N in [100, 10]:
        for mu in ['1e-06', '1e-04', '1e-02']:
            graphData = []
            for i in range(len(algs)):
                graphData.append([])
                reader = csv.reader(open(path + "/iterates_" + algs[i] + "_N=" + str(N) + "_mu=" + mu + "_n=" + n + ".csv"), delimiter=",")
                for _ in range(numTests):
                    x = next(reader)
                    y = next(reader)
                    graphData[i].append([float(x_) for x_ in x])
                    graphData[i].append([float(y_) for y_ in y])

            data.append({
                'mu': mu,
                'N': N,
                'graph': graphData
                })

    allData[n] = data

plt.rcParams.update({'font.size': 14})
plt.rc('text', usetex=True)

for n in n_list:
    if n == '1000':
        numTests = 100
        algs = ['WLK21(ErrorCR)', 'WLK21']
        colors = ['#FF7F00', '#984EA3']
        xlims = [8, 3.5, 1, 1.5, 0.8, 0.3]
        lw, a = 1, 0.25
        filename = '1e3fig_ErrorCR.pdf'
    elif n == '10000':
        numTests = 100
        algs = ['WLK21(ErrorCR)', 'WLK21']
        colors = ['#FF7F00', '#984EA3']
        xlims = [300, 150, 40, 40, 20, 6]
        lw, a = 1, 0.25
        filename = '1e4fig_ErrorCR.pdf'
    elif n == '100000':
        numTests = 5
        algs = ['WLK21(ErrorCR)', 'WLK21']
        colors = ['#FF7F00', '#984EA3']
        xlims = [8000,4000,1000,500,300,60]
        lw, a = 1, 1
        filename = '1e5fig_ErrorCR.pdf'

    numAlgs = len(algs)
    fig, axs = plt.subplots(2, 3)

    data = allData[n]

    for graph in range(6):
        graphData = data[graph]['graph']
        title = r'$\bar\mu^* = ' 
        if data[graph]['mu'] == '1e-02':
            title += '10^{-2}'
        elif data[graph]['mu'] == '1e-04':
            title += '10^{-4}'
        else:
            title += '10^{-6}'
        if data[graph]['N'] == 10:
            title += r',\,\bar N = 10^{' + str(len(n)) + r'}$'
        else:
            title += r',\,\bar N = 10^{' + str(len(n) + 1) + r'}$'

        for algNum, algData in enumerate(graphData):
            for i in range(numTests):
                x = np.array(algData[2 * i])
                y = np.array(algData[2 * i + 1])
                x = np.extract(y > 0, x)
                y = np.extract(y > 0, y)
                # print(all(y > 0))
                axs[1 - graph // 3, 2 - graph % 3].plot(x, y, color=colors[algNum], linewidth=lw, alpha=a)

        axs[1 - graph // 3, 2 - graph % 3].set_yscale('log')
        axs[1 - graph // 3, 2 - graph % 3].set_title(title)
        axs[1 - graph // 3, 2 - graph % 3].grid()
        axs[1 - graph // 3, 2 - graph % 3].set_xlim([0,xlims[graph]])
        axs[1 - graph // 3, 2 - graph % 3].set_ylim([1e-14, 1e0])
        axs[1 - graph // 3, 2 - graph % 3].set_xlabel('time (s)')
        axs[1 - graph // 3, 2 - graph % 3].set_ylabel('error')

    patches = []
    for algNum, algName in enumerate(algs):
        patches.append(mpatches.Patch(color=colors[algNum], label=algName))
    patches.reverse()

    fig.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, right=1-0.04, top=1-0.075, wspace=0.35, hspace=0.45)
    fig.legend(handles=patches, loc="lower center", ncol=numAlgs)

    fig.set_size_inches(14,7)
    plt.savefig(path_results + filename)


###################

# make tables

Tables = []
mu2str = {'0.01': '1e-2', '0.0001': '1e-4', '1e-06': '1e-6'}

for n in n_list:
    if n == '1000':
        path = path_outputs + "results_server_1e3/"
        densities = ['0.01', '0.1']
    elif n == '10000':
        path = path_outputs + "results_server_1e4/"
        densities = ['0.001', '0.01']
    elif n == '100000':
        path = path_outputs + "results_server_1e5/"
        densities = ['0.0001', '0.001']

    data = {}

    for mu in ['0.01', '0.0001', '1e-06']:
        if n == '1000':
            table = [
                [None, 'WLK21'],
                [None, 'WK20'],
                [mu2str[mu], 'JL19'],
                [None, 'AN19'],
                [None, 'BTH14']]
        elif n == '10000':
            table = [
                [None, 'WLK21'],
                [None, 'WK20'],
                [mu2str[mu], 'JL19'],
                [None, 'AN19']]
        elif n == '100000':
            table = [
                [None, 'WLK21'],
                [mu2str[mu], 'WK20'],
                [None, 'JL19']]

        for density in densities:
            path1 = 'avg_density=' + density + '_mu=' + mu + '_n=' + n + '.csv'
            path2 = 'avg_eig_density=' + density + '_mu=' + mu + '_n=' + n + '.csv'
            data_avg = pd.read_csv(path + path1).values
            data_times = pd.read_csv(path + path2).values

            if n == '1000':
                table[0] = table[0] + [data_avg[0][2]] + [data_avg[2][2]] + [data_avg[0][1]] + [data_times[0][1]] + [data_times[0][2]]
                table[1] = table[1] + [data_avg[1][2]] + [data_avg[3][2]] + [data_avg[1][1]] + [data_times[1][1]] + [data_times[1][2]]
                table[2] = table[2] + [data_avg[6][2]] + [data_avg[7][2]] + [data_avg[6][1]] + [data_times[2][1]] + [data_times[2][2]]
                table[3] = table[3] + [data_avg[4][2]] + [None] + [data_avg[4][1]] + [None, None]
                table[4] = table[4] + [data_avg[5][2]] + [None] + [data_avg[5][1]] + [None, None]
            elif n == '10000':
                table[0] = table[0] + [data_avg[0][2]] + [data_avg[2][2]] + [data_avg[0][1]] + [data_times[0][1]] + [data_times[0][2]]
                table[1] = table[1] + [data_avg[1][2]] + [data_avg[3][2]] + [data_avg[1][1]] + [data_times[1][1]] + [data_times[1][2]]
                table[2] = table[2] + [data_avg[5][2]] + [data_avg[6][2]] + [data_avg[5][1]] + [data_times[2][1]] + [data_times[2][2]]
                table[3] = table[3] + [data_avg[4][2]] + [None] + [data_avg[4][1]] + [None, None]
            elif n == '100000':
                table[0] = table[0] + [data_avg[0][2]] + [data_avg[2][2]] + [data_avg[0][1]] + [data_times[0][1]] + [data_times[0][2]]
                table[1] = table[1] + [data_avg[1][2]] + [data_avg[3][2]] + [data_avg[1][1]] + [data_times[1][1]] + [data_times[1][2]]
                table[2] = table[2] + [data_avg[4][2]] + [data_avg[5][2]] + [data_avg[4][1]] + [data_times[2][1]] + [data_times[2][2]]

        data[float(mu)] = table
    
    Tables.append(data)

files = ['table3.tex' , 'table4.tex', 'table5.tex']
keys = [1e-2, 1e-4, 1e-6]

for data, file in zip(Tables, files):
    f = open(path_results + file, 'w')
    print("\\begin{tabular}{ccccccc|ccccc}%", file=f)
    print("\\toprule", file=f)

    print(file)
    if file == files[0]:
        print("& & \\multicolumn{5}{c}{$\\bar N = 10^4$} & \\multicolumn{5}{c}{$\\bar N = 10^5$} \\\\", file=f)
    elif file == files[1]:
        print("& & \\multicolumn{5}{c}{$\\bar N = 10^5$} & \\multicolumn{5}{c}{$\\bar N = 10^6$} \\\\", file=f)
    elif file == files[2]:
        print("& & \\multicolumn{5}{c}{$\\bar N = 10^6$} & \\multicolumn{5}{c}{$\\bar N = 10^7$} \\\\", file=f)

    print("\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}", file=f)
    print("& & & & & \\multicolumn{2}{c}{Time} & & & & \\multicolumn{2}{c}{Time} \\\\", file=f)
    print("\\cmidrule(lr){6-7} \\cmidrule(lr){11-12}", file=f)
    print("$\\bar\\mu^*$   & Alg.        & \\texttt{Error} & \\texttt{ErrorCR} & Time      & Ref.\\ & Solve & \\texttt{Error} & \\texttt{ErrorCR} & Time     & Ref.\\  & Solve \\\\", file=f)

    for key in keys:
        print ('\\midrule', file=f)
        for row in data[key]:
            # size
            if row[0] == None:
                print('', end='', file=f)
            else:
                print(row[0], end='', file=f)
            # name
            print('& \\texttt{', row[1], '}', end='', file=f)
            # error
            if 1<= row[2] * 1e16 and row[2] * 1e16 < 10:
                print('& ', "{:.1f}".format(row[2] * 1e16), end='', file=f)
            else:
                print('& ', "{:.1e}".format(row[2] * 1e16), end='', file=f)
            # errorCR
            if row[3] == None:
                print('& -', end='', file=f)
            else:
                if 1<= row[3] * 1e16 and row[3] * 1e16 < 10:
                    print('& ', "{:.1f}".format(row[3] * 1e16), end='', file=f)
                else:
                    print('& ', "{:.1e}".format(row[3] * 1e16), end='', file=f)
            # time
            if row[4] == min([r[4] for r in data[key]]):
                print('& \\textbf{', "{:.1f}".format(row[4]), '}', end='', file=f)
            else:
                print('& ', "{:.1f}".format(row[4]), end='', file=f)
            # reformulate
            if row[5] == None:
                print('& -', end='', file=f)
            else:
                print('& ', "{:.1f}".format(row[5]), end='', file=f)
            # solve
            if row[6] == None:
                print('& -', end='', file=f)
            else:
                print('& ', "{:.1f}".format(row[6]), end='', file=f)
            # error
            if 1<= row[7] * 1e16 and row[7] * 1e16 < 10:
                print('& ', "{:.1f}".format(row[7] * 1e16), end='', file=f)
            else:
                print('& ', "{:.1e}".format(row[7] * 1e16), end='', file=f)
            # errorCR
            if row[8] == None:
                print('& -', end='', file=f)
            else:
                if 1<= row[8] * 1e16 and row[8] * 1e16 < 10:
                    print('& ', "{:.1f}".format(row[8] * 1e16), end='', file=f)
                else:
                    print('& ', "{:.1e}".format(row[8] * 1e16), end='', file=f)
            # time
            if row[9] == min([r[9] for r in data[key]]):
                print('& \\textbf{', "{:.1f}".format(row[9]), '}', end='', file=f)
            else:
                print('& ', "{:.1f}".format(row[9]), end='', file=f)
            # reformulate
            if row[10] == None:
                print('& -', end='', file=f)
            else:
                print('& ', "{:.1f}".format(row[10]), end='', file=f)
            # solve
            if row[11] == None:
                print('& -', '\\\\', file=f)
            else:
                print('& ', "{:.1f}".format(row[11]), '\\\\', file=f)
                
    print('\\bottomrule', file=f)
    print('\\end{tabular}', file=f)

