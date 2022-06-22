import numpy as np
import os
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize


def visualize_forgetting_events(data_path):
    '''Visualizes forgetting events numpy array in data_path
    Parameters:
        :param data_path: path to the example forgetting statistics
        :type data_path: str'''

    # get names
    learned_name = data_path + 'learned.npy'
    forgetting_name = data_path + 'forgetting_events.npy'

    # load files
    forgetting_events = np.load(forgetting_name)
    learned = np.load(learned_name)

    # init plot array
    maxi = max(forgetting_events)
    visualization = np.zeros(maxi + 1, dtype=int)
    for sample in forgetting_events:
        visualization[sample] += 1

    fraction = float(visualization[0])/float(forgetting_events.shape[0])
    print('Fraction Unforgettable: %f' % fraction)
    # plot histogram
    plt.hist(forgetting_events, density=False, bins='auto')
    plt.ylabel('Number of samples')
    plt.xlabel('#Forgetting Events')
    plt.show()


def visualize_example_forgetting_sequence(folder_list, index_list, name='forgetting_events', prefix='',
                                          add_distributions=False):
    ''' Visualizes forgetting events transition during active learning.
    Parameters:
        :param folder_list: list of folders that contatin the different query strategy npy files.
        :type folder_list: list
        :param index_list: list of different rounds that should be visualized
        :type index_list: list'''
    # init size of plot
    types = len(folder_list)
    amount = len(index_list)

    if add_distributions:
        rows = types * 2
    else:
        rows = types

    # init subplots

    fig, axs = plt.subplots(rows, amount, sharex=False)
    fig.tight_layout()
    # iterate through lists
    for i in range(types):
        for j in range(amount):
            # load array
            forgetting_name = folder_list[i] + prefix + name + str(index_list[j]) + '.npy'
            forgetting_array = np.load(forgetting_name)
            if add_distributions:
                # init gmm and predict samples
                gmm = GaussianMixture(n_components=2, n_init=20).fit(forgetting_array.reshape(-1, 1))
                #gmm = KMeans(n_clusters=2, random_state=0).fit(forgetting_array.reshape(-1, 1))
                xrange = np.arange(0, max(forgetting_array), 0.1)
                pred = gmm.predict(forgetting_array.reshape(-1, 1))
                #pred = gmm.labels_
                samples1_inds = (pred == 0).nonzero()
                samples2_inds = (pred == 1).nonzero()
                mean1 = gmm.means_[0][0]
                #mean1 = np.mean(forgetting_array[samples1_inds[0]])
                cov1 = gmm.covariances_[0]
                mean2 = gmm.means_[1][0]
                #mean2 = np.mean(forgetting_array[samples2_inds[0]])
                cov2 = gmm.covariances_[0]
                print('Statistics of Round %d and Type %d' % (j, i))
                print(mean1)
                print(mean2)
                print(cov1)
                print(cov2)
                dist1 = np.array([normpdf(x, mean1, cov1) for x in xrange])
                dist2 = np.array([normpdf(x, mean2, cov2) for x in xrange])

            # remove largest element since it is either sochastically irrelevant or never learned
            forgetting_array = forgetting_array[forgetting_array != max(forgetting_array)]
            total = forgetting_array.shape[0]

            if types != 1:
                # plot histograms
                if i == 0:
                    # set title only in first row
                    axs[i, j].set_title('Round: ' + str(index_list[j]) + ' Total Samples: ' + str(total))
                if add_distributions:
                    axs[i*2, j].hist(forgetting_array, density=False, bins='auto')
                    axs[i * 2 + 1, j].plot(dist1)
                    axs[i * 2 + 1, j].plot(dist2)
                else:
                    axs[i, j].hist(forgetting_array, density=False, bins='auto')

                # set axis names only in specifc rows
                if j == 0:
                    axs[i, j].set(xlabel='#Forgetting Events', ylabel='Number of samples')
                else:
                    axs[i, j].set(xlabel='#Forgetting Events')
            else:
                # plot histograms
                if add_distributions:
                    axs[i*2, j].hist(forgetting_array, density=False, bins='auto')
                    axs[i * 2 + 1, j].plot(dist1)
                    axs[i * 2 + 1, j].plot(dist2)
                else:
                    axs[j].set_title('Round ' + str(index_list[j]))
                    axs[j].hist(forgetting_array, density=False, bins='auto')
                    # set axis names only in specifc rows
                    if j == 0:
                        axs[j].set(xlabel='#Forgetting Events', ylabel='Number of samples')
                    else:
                        axs[j].set(xlabel='#Forgetting Events')
    plt.show()


def get_dataframe(target_path):
    """
    Returns aggregated dataframe with all runs stored with and additional column id
    Parameters:
        :param target_path: path to folder with target excel files
        :type target_path: str
    """

    # get all excel files
    path_list = glob.glob(target_path + '*')
    total = pd.DataFrame([])

    # add excel files into complete dataframe
    for i in range(len(path_list)):
        if path_list[i].endswith('.xlsx'):
            df = pd.read_excel(path_list[i], index_col=0, engine='openpyxl')
            df['Trial'] = i + 1
            df['Round'] = range(0, df.shape[0])
            total = pd.concat([total, df])
    return total


def kermany_patient_dist(split_file):
    import os
    from collections import Counter

    fullname = os.path.join('./Data/Kermany', split_file + '.txt')
    files = open(fullname, "r")
    filelist = files.readlines()
    filelist = [id_.rstrip().split('-') for id_ in filelist]
    filelist = np.asarray(filelist)
    patients = filelist[:, 1]
    counts = Counter(patients.tolist())
    class_patients = dict()
    for j, id in enumerate(filelist[:, 1]):
        if id not in class_patients.keys():
            class_patients[id] = filelist[j, 0]

    df = pd.DataFrame([counts.keys(), class_patients.values(), counts.values()])
    df = df.transpose()
    df.columns = ['ID', 'class', 'count']
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    t1 = df.loc[df['class'] == 'CNV']
    t2 = df.loc[df['class'] == 'DME']
    t3 = df.loc[df['class'] == 'DRUSEN']
    plt.bar(t1['ID'], t1['count'], color='b', label='CNV: ' + str(t1.shape[0]) + ' Patients')
    plt.bar(t2['ID'], t2['count'], color='g', label='DME: ' + str(t2.shape[0]) + ' Patients')
    plt.bar(t3['ID'], t3['count'], color='r', label='DRUSEN: ' + str(t3.shape[0]) + ' Patients')

    plt.xlabel('Patient ID\'s')
    plt.ylabel('Image Count')
    plt.title('Image Distribution Across Patients in OCT ' + split_file + ' Set')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()


def kermany_xray_patient_dist(split_file):
    import os
    from collections import Counter
    path = '../../../../../OCT/BIGandDATA/ZhangData/chest_xray'
    classes = ['NORMAL', 'PNEUMONIA']
    files_temp = os.listdir(os.path.join(path, split_file, classes[0]))

    files = open(os.path.join('./Data/Kermany_xray', split_file + '.txt'), "r")
    filelist = files.readlines()
    filelist = [id_.rstrip().split('-') for id_ in filelist]

    filelist1 = np.asarray(filelist[:len(files_temp)])
    files1 = np.asarray(filelist1)
    filelist2 = np.asarray(filelist[len(files_temp):])
    files2 = np.asarray(filelist2)
    files2 = np.delete(files2, 0, 1)

    all_files = np.concatenate((files1, files2), axis=0)
    patients = all_files[:, 1]
    counts = Counter(patients.tolist())
    class_patients = dict()
    for j, id in enumerate(all_files[:, 1]):
        if id not in class_patients.keys():
            class_patients[id] = all_files[j, 0]
    df = pd.DataFrame([counts.keys(), class_patients.values(), counts.values()])
    df = df.transpose()
    df.columns = ['ID', 'class', 'count']

    t1 = df.loc[df['class'] == 'NORMAL']
    t2 = df.loc[df['class'] == 'VIRUS']
    t3 = df.loc[df['class'] == 'BACTERIA']
    plt.bar(t1['ID'], t1['count'], color='b', label='NORMAL: ' + str(t1.shape[0]) + ' Patients')
    plt.bar(t2['ID'], t2['count'], color='g', label='VIRUS: ' + str(t2.shape[0]) + ' Patients')
    plt.bar(t3['ID'], t3['count'], color='r', label='BACTERIA: ' + str(t3.shape[0]) + ' Patients')

    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    plt.xlabel('Patient ID\'s')
    plt.ylabel('Image Count')
    plt.title('Image Distribution Across Patients in X-Ray ' + split_file + ' Set')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()


def visualize_forgetting_events_confusion_matrix(file_path, display_samples=None, name='forgetting_events', prefix=''):
    # get all files in path
    files = glob.glob(file_path + prefix + name + '*')

    # iterate over files and stack forgetting events
    first = True
    for i in range(len(files)):
        fevent_array = np.load(file_path + prefix + name + str(i) + '.npy')
        maxi = fevent_array.max()
        fevent_array[fevent_array == maxi] = 0
        fevent_array = fevent_array.astype(float)/maxi
        # create stack array if first run
        if first:
            first = False
            if display_samples is not None:
                stack = np.empty((0, display_samples))
            else:
                stack = np.empty((0, fevent_array.shape[0]))

        # append fevent array to stack array
        # if display_samples is not None:
        #     stack = np.concatenate((stack, fevent_array[np.newaxis, :display_samples]), axis=0)
        # else:
        #     stack = np.concatenate((stack, fevent_array[np.newaxis, :]), axis=0)

    # visualize stack
    im = plt.imshow(stack, cmap='jet', aspect='auto')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Sample Index', fontsize=13)
    plt.ylabel('Time', fontsize=13)
    plt.colorbar(im)
    plt.show()


def visualize_patient_accuracy_confusion_matrix(file_path, display_samples=None, name='test_accuracy', prefix='patient_'):
    # get all files in path
    files = glob.glob(file_path + prefix + name + '*')

    # iterate over files and stack forgetting events
    first = True
    for i in range(len(files)):
        # fevent_array = np.load(file_path + prefix + name + str(i) + '.xlsx')
        df = pd.read_excel(file_path + prefix + name + str(i) + '.xlsx', engine='openpyxl')
        accuracy = df['Predicted Correct'] / df['Total Count']  # patient accuracy
        # label = df['Ground Truth']

        if first:
            first = False
            if display_samples is not None:
                stack = np.empty((0, display_samples))
                # stack2 = np.empty((0, display_samples))
            else:
                stack = np.empty((0, accuracy.shape[0]))
                # stack2 = np.empty((0, label.shape[0]))

        # append fevent array to stack array
        if display_samples is not None:
            stack = np.concatenate((stack, accuracy[np.newaxis, :display_samples]), axis=0)
            # stack2 = np.concatenate((stack, label[np.newaxis, :display_samples]), axis=0)
        else:
            stack = np.concatenate((stack, accuracy[np.newaxis, :]), axis=0)
            # stack2 = np.concatenate((stack, label[np.newaxis, :]), axis=0)

    # visualize stack
    im = plt.imshow(stack, cmap='jet', aspect='auto', origin='lower', extent=[0, len(accuracy), 0, len(files)])
    plt.yticks(np.arange(0, len(files), dtype=np.int))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Patient ID\'s', fontsize=13)
    plt.ylabel('Round', fontsize=13)
    plt.title('Patient Accuracy Across ' + file_path.split('/')[-2] + ' Rounds in Testset')
    plt.colorbar(im)
    plt.show()


def patient_intersection_train_test():
    # The function confirms there is no pverlap between patients in the train and test sets
    import os
    from collections import Counter

    trainfiles = open(os.path.join('./Data/Kermany/train.txt'), "r")
    testfiles = open(os.path.join('./Data/Kermany/test.txt'), "r")
    trainlist = trainfiles.readlines()
    testlist = testfiles.readlines()
    trainlist = [id_.rstrip().split('-') for id_ in trainlist]
    testlist = [id_.rstrip().split('-') for id_ in testlist]
    trainlist = np.asarray(trainlist)
    testlist = np.asarray(testlist)
    train_patients = trainlist[:, 1]
    test_patients = testlist[:, 1]
    train_counts = Counter(train_patients.tolist())
    test_counts = Counter(test_patients.tolist())

    overlap = {}
    for patient in train_counts.keys():
        if patient in test_counts.keys():
            # The number of times the patients is in the test set
            overlap[patient] = test_counts[patient]
        else:
            overlap[patient] = 0

    indices = np.arange(len(train_counts.keys()))
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    # plt.bar(indices, train_counts.values(), color='b', label='Patients in trainset')
    plt.bar(indices, overlap.values(), color='r', alpha=0.5)
    plt.xlabel('Patient ID\'s')
    plt.ylabel('Samples')
    plt.title('Patient overlap between train and test set')
    plt.show()


def plot_lc(path_list, names, plotting_col='Test Acc', ci=None, xlim=None, ylim=None):

    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_dataframe(target_path=path_list[i])
        df['Algorithm'] = names[i]
        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])

    sns.lineplot(data=total, x='Samples', y=plotting_col, hue='Algorithm', ci=ci)
    # sns.displot(total, x=plotting_col, hue='Algorithm', kind='kde')
    # plt.grid()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc="lower right", prop={'size': 10})
    # plt.legend(loc="best", prop={'size': 10})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Samples', fontsize=13)
    arch = path_list[0].split('/')[2]

    if arch == 'resnet_18':
        title = 'Resnet 18'
    elif arch == 'resnet_50':
        title = 'Resnet 50'
    elif arch == 'vgg_16':
        title = 'VGG 16'
    elif arch == 'densenet_121':
        title = 'Densenet 121'

    plt.title(title, fontsize=13)
    plt.ylabel('Test Accuracy', fontsize=13)
    plt.show()


def vis_variance_per_trial(path_list, names):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(len(path_list)):
        df = get_dataframe(target_path=path_list[i])
        df['Sampling'] = names[i]
        sns.boxplot(x="Sampling", y="Test Acc",
                    hue="Trial", palette=colors,
                    data=df)
        plt.title(path_list[i].split('/')[2])
        plt.show()


def vis_compare_variance_per_trial(path1, path2, names):
    if len(path1) != len(path2):
        raise Exception("Lengths of path lists must be the same.")
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(len(path1)):
        df1 = get_dataframe(target_path=path1[i])
        df2 = get_dataframe(target_path=path2[i])
        df1['Sampling'] = names[i]
        df1['Initialization Strategy'] = 'random initialization'
        df2['Initialization Strategy'] = 'diverse initialization'
        df2['Sampling'] = names[i]
        DF = pd.concat([df1, df2])
        sns.boxplot(x="Initialization Strategy", y="Test Acc",
                    hue="Trial", palette=colors,
                    data=DF)
        arch = path1[i].split('/')[2]
        if arch == 'resnet_18':
            title = 'Resnet 18'
        elif arch == 'resnet_50':
            title = 'Resnet 50'
        elif arch == 'vgg_16':
            title = 'VGG 16'
        elif arch == 'densenet_121':
            title = 'Densenet 121'

        plt.title(title + ' - ' + names[i])
        plt.grid()
        plt.show()


def vis_strategy_variance_per_trial(std_paths, div_paths, standard_names, diverse_names):
    if len(std_paths) != len(div_paths):
        raise Exception("Lengths of path lists must be the same.")
    # colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(len(std_paths)):
        df1 = get_dataframe(target_path=std_paths[i])
        df2 = get_dataframe(target_path=div_paths[i])
        df1['Sampling'] = standard_names[i]
        # df1['Initialization Strategy'] = 'random initialization'
        # df2['Initialization Strategy'] = 'diverse initialization'
        df2['Sampling'] = diverse_names[i]
        # df1 = df1.loc[:, "Test Acc"].mean() #.nlargest(3)
        # df2 = df2.loc[:, "Test Acc"].mean() #.nlargest(3)
        DF = pd.concat([df1, df2])
        sns.boxplot(x="Sampling", y="Test Acc",
                    hue="Trial", palette=sns.color_palette("bright"),
                    data=DF)
        arch = std_paths[i].split('/')[2]
        if arch == 'resnet_18':
            title = 'Resnet 18'
        elif arch == 'resnet_50':
            title = 'Resnet 50'
        elif arch == 'vgg_16':
            title = 'VGG 16'
        elif arch == 'densenet_121':
            title = 'Densenet 121'

        plt.title(title + ' - ' + standard_names[i] + ' vs ' + diverse_names[i])
        plt.grid()
        plt.show()


def vis_variance_per_round(path_list, names, rounds):
    if len(path_list) != len(names):
        raise Exception("Each element in the path_list must have a corresponding name")
    total = pd.DataFrame([])
    for i in range(len(path_list)):
        df = get_dataframe(target_path=path_list[i])
        df['Sampling'] = names[i]
        total = pd.concat([total, df])
    for name in names:
        # for round in rounds:
        DF = pd.DataFrame([])
        DF = pd.concat([DF, total.loc[total["Sampling"] == name]])
        DF = DF.loc[rounds]
        sns.boxplot(x="Sampling", y="Test Acc", hue='Round', palette=sns.color_palette("bright"), data=DF)
        plt.title(path_list[i].split('/')[2])
        plt.show()
        a = 1


def vis_compare_variance_per_round(path1, path2, names, rounds):
    if len(path1) != len(path2):
        raise Exception("Lengths of path lists must be the same.")
    total = pd.DataFrame([])
    for i in range(len(path1)):
        df1 = get_dataframe(target_path=path1[i])
        df2 = get_dataframe(target_path=path2[i])
        df1['Sampling'] = names[i]
        df1['Initialization Strategy'] = 'random initialization'
        df2['Initialization Strategy'] = 'diverse initialization'
        df2['Sampling'] = names[i]
        DFF = pd.concat([df1, df2])
        total = pd.concat([total, DFF])
    for name in names:

        # for round in rounds:
        DF = pd.DataFrame([])
        DF = pd.concat([DF, total.loc[total["Sampling"] == name]])
        if not DF.empty:
            DF = DF.loc[rounds]
            sns.boxplot(x="Initialization Strategy", y="Test Acc", hue='Round', palette=sns.color_palette("bright"), data=DF)
            print(name)
            print("round = " + str(rounds[0]))
            rand_init = DF['Initialization Strategy'] == 'random initialization'
            diverse_init = DF['Initialization Strategy'] == 'diverse initialization'
            print("mean rand init - top 3 " + str(DF.loc[rand_init, "Test Acc"].nlargest(3).mean()))
            print("mean diverse init - top 3 " + str(DF.loc[diverse_init, "Test Acc"].nlargest(3).mean()))
            print("mean rand init - all 5 " + str(DF.loc[rand_init, "Test Acc"].mean()))
            print("mean diverse init - all 5 " + str(DF.loc[diverse_init, "Test Acc"].mean()) + '\n\n')
            arch = path1[i].split('/')[2]
            if arch == 'resnet_18':
                title = 'Resnet 18'
            elif arch == 'resnet_50':
                title = 'Resnet 50'
            elif arch == 'vgg_16':
                title = 'VGG 16'
            elif arch == 'densenet_121':
                title = 'Densenet 121'

            plt.title(title + ' - ' + name)

            # plt.show()


def compare_initializations(arch, dataset, init, nstart, pretrained=False):
    dir = 'initializations' + str(nstart) + '_pretrained' if pretrained else 'initializations' + str(nstart)
    path = os.path.join('output', dataset, arch, init, dir)
    average = 0
    all_accuracy = []

    for i in range(1, 6):
        fullpath = os.path.join(path, 'roundzero_' + str(i))
        files = os.listdir(fullpath)
        if 'patient_test_accuracy0.xlsx' in files:
            df = pd.read_excel(os.path.join(fullpath, 'patient_test_accuracy0.xlsx'), index_col=0, engine='openpyxl')
            correct_pred = df['Predicted Correct'].sum()
            total_count = df['Total Count'].sum()
            acc = (correct_pred/total_count) * 100
            print('accuracy ' + str(i) + ': ' + str(acc))
            # df['Round'] = range(0, df.shape[0])
            all_accuracy.append(acc)
            average += acc
    all_accuracy.sort()
    top3 = sum(all_accuracy[-3:])
    print('average = ' + str(average / 5))
    print('average top 3 = ' + str(top3 / 3))

def compare_NFR(arch, dataset, query, metric):
    total = pd.DataFrame([])
    root = os.path.join('output', dataset, arch, 'diverse_init', 'AttGuided')
    for i in range(1, 5):
        # path1 = os.path.join(root, metric, query[0] + str(i) +'_128')
        # path2 = os.path.join(root, metric, query[1] + str(i) +'_128')
        path1 = os.path.join(root, query[0] + str(i) +'_64')
        path2 = os.path.join(root, query[1] + str(i) +'_64')
        df1 = pd.DataFrame([])
        df2 = pd.DataFrame([])
        for j in range(44):
            data_old1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
            data_new1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
            y_old1 = data_old1['prediction'] == data_old1['GT']
            y_new1 = data_new1['prediction'] != data_new1['GT']
            NFR1 = (y_old1 & y_new1).mean()
            df1.loc[j, 'strategy'] = query[0] 
            df1.loc[j, 'NFR'] = NFR1
            df1.loc[j, 'Round'] = j
        for j in range(44):
            data_old2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
            data_new2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
            y_old2 = data_old2['prediction'] == data_old2['GT']
            y_new2 = data_new2['prediction'] != data_new2['GT']
            NFR2 = (y_old2 & y_new2).mean()
            df2.loc[j, 'strategy'] = query[1]
            df2.loc[j, 'NFR'] = NFR2
            df2.loc[j, 'Round'] = j
        total = pd.concat([total, df1, df2])

    sns.lineplot(x="Round", y="NFR", hue='strategy', data=total, ci=95)
    if arch == 'resnet_18':
        title = 'Resnet 18'
    elif arch == 'resnet_50':
        title = 'Resnet 50'
    elif arch == 'vgg_16':
        title = 'VGG 16'
    elif arch == 'densenet_121':
        title = 'Densenet 121'

    plt.title(title + ' ' + dataset, fontsize=13)
    plt.ylabel('NFR', fontsize=13)
    plt.xlabel('Round', fontsize=13)

    plt.show()

# original
def compare_NFR_traditional_to_PCT_original(arch, dataset, query):
    total = pd.DataFrame([])
    root = os.path.join('output', dataset, arch, 'diverse_init')

    for i in range(1, 3):
        path1 = os.path.join(root, 'NFR', query + str(i) +'_128')
        path2 = os.path.join(root, 'PCT', query + str(i) +'_128')
        df1 = pd.DataFrame([])
        df2 = pd.DataFrame([])
        for j in range(20):
            data_old1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
            data_new1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
            y_old1 = data_old1['prediction'] == data_old1['GT']
            y_new1 = data_new1['prediction'] != data_new1['GT']
            NFR1 = (y_old1 & y_new1).mean()
            df1.loc[j, 'train type'] = 'Traditional'
            df1.loc[j, 'strategy'] = query
            df1.loc[j, 'NFR'] = NFR1
            df1.loc[j, 'Round'] = j
        for j in range(20):
            data_old2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
            data_new2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
            y_old2 = data_old2['prediction'] == data_old2['GT']
            y_new2 = data_new2['prediction'] != data_new2['GT']
            NFR2 = (y_old2 & y_new2).mean()
            df2.loc[j, 'train type'] = 'Positive Congruent'
            df2.loc[j, 'strategy'] = query
            df2.loc[j, 'NFR'] = NFR2
            df2.loc[j, 'Round'] = j

        total = pd.concat([total, df1, df2])

    sns.lineplot(x="Round", y="NFR", hue='train type', data=total, ci=95)
    if arch == 'resnet_18':
        title = 'Resnet 18'
    elif arch == 'resnet_50':
        title = 'Resnet 50'
    elif arch == 'vgg_16':
        title = 'VGG 16'
    elif arch == 'densenet_121':
        title = 'Densenet 121'

    plt.title(title + ' ' + dataset + ' ' + query, fontsize=13)
    plt.ylabel('NFR', fontsize=13)
    plt.xlabel('Round', fontsize=13)

    # plt.show()
    save_here = os.path.join('excel', dataset.split('Patient')[0], arch, 'diverse_init', 'figures', 'NFR')
    if not os.path.exists(save_here):
        os.makedirs(save_here)
    plt.savefig(os.path.join(save_here, title + ' ' + dataset + ' ' + query + '.png'))
    plt.clf()


# delete this after
def compare_NFR_traditional_to_PCT(arch, dataset, query):
    total = pd.DataFrame([])
    # root = os.path.join('output', dataset, arch, 'diverse_init') # kermany OFFICIAL
    # root = os.path.join('output_default', dataset, arch, 'diverse_init')  # kermany
    # root = os.path.join('output_default', dataset, arch, 'rand_init')  # kermany
    root = os.path.join('output_default', dataset, arch, 'rand_init')  # cifar-10

    for i in range(1, 2):
        # path1 = os.path.join(root, 'NFR', query + str(i) +'_128')
        # path2 = os.path.join(root, 'PCT', query + str(i) +'_128')
        path2 = os.path.join(root, 'output_default')
        # df1 = pd.DataFrame([])
        df2 = pd.DataFrame([])
        # for j in range(20):
        #     data_old1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
        #     data_new1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
        #     y_old1 = data_old1['prediction'] == data_old1['GT']
        #     y_new1 = data_new1['prediction'] != data_new1['GT']
        #     NFR1 = (y_old1 & y_new1).mean()
        #     df1.loc[j, 'train type'] = 'Traditional'
        #     df1.loc[j, 'strategy'] = query
        #     df1.loc[j, 'NFR'] = NFR1
        #     df1.loc[j, 'Round'] = j
        # df1['mov_avg'] = df1['NFR'].rolling(4).mean()
        for j in range(4):
            data_old2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
            data_new2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
            y_old2 = data_old2['prediction'] == data_old2['GT']
            y_new2 = data_new2['prediction'] != data_new2['GT']
            # print(sum(y_old2 & y_new2))

            NFR2 = (y_old2 & y_new2).mean()
            print(NFR2)
            df2.loc[j, 'train type'] = 'Positive Congruent'
            df2.loc[j, 'strategy'] = query
            df2.loc[j, 'NFR'] = NFR2
            df2.loc[j, 'Round'] = j
        # df2['mov_avg'] = df2['NFR'].rolling(4).mean()

        total = pd.concat([total, df2])

    # df.loc[df['class'] == 'CNV']
    # trad = total.loc[total['train type'] == 'Traditional', 'mov_avg']
    # pc = total.loc[total['train type'] == 'Positive Congruent', 'mov_avg']
    # diff = (trad - pc).abs().sum()
    #
    #
    # print(query + ': ' + str(diff))

    sns.lineplot(x="Round", y="NFR", hue='train type', data=total, ci=95)
    if arch == 'resnet_18':
        title = 'Resnet 18'
    elif arch == 'resnet_50':
        title = 'Resnet 50'
    elif arch == 'vgg_16':
        title = 'VGG 16'
    elif arch == 'densenet_121':
        title = 'Densenet 121'

    plt.title(title + ' ' + dataset + ' ' + query, fontsize=13)
    plt.ylabel('NFR', fontsize=13)
    plt.xlabel('Round', fontsize=13)

    plt.show()


def compare_accuracy_traditional_to_PCT(arch, dataset, xlim=None):
    root1 = os.path.join('excel', dataset.split('Patient')[0], arch, 'diverse_init', 'NFR')
    root2 = os.path.join('excel', dataset.split('Patient')[0], arch, 'diverse_init', 'PCT')

    folders = ['rand_128', 'badge_128', 'entropy_128', 'least_conf_128', 'margin_128', 'patient_diverse_128',
               'patient_diverse_badge_128', 'patient_diverse_entropy_128', 'patient_diverse_least_conf_128',
               'patient_diverse_margin_128']

    for i in range(len(folders)):
        df1 = get_dataframe(target_path=os.path.join(root1, folders[i] + '/'))
        df2 = get_dataframe(target_path=os.path.join(root2, folders[i] + '/'))
        df1['Sampling'] = folders[i].split('_128')[0]
        df2['Sampling'] = folders[i].split('_128')[0]
        df1['train type'] = 'Traditional'
        df2['train type'] = 'Positive Congruent'
        DF = pd.concat([df1, df2])

        sns.lineplot(x="Samples", y="Test Acc",
                     hue="train type", ci=60,
                     data=DF)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])

        # arch = std_paths[i].split('/')[2]
        if arch == 'resnet_18':
            title = 'Resnet 18'
        elif arch == 'resnet_50':
            title = 'Resnet 50'
        elif arch == 'vgg_16':
            title = 'VGG 16'
        elif arch == 'densenet_121':
            title = 'Densenet 121'

        plt.title(title + ' - ' + folders[i].split('_128')[0])
        # plt.show()
        save_here = os.path.join('excel', dataset.split('Patient')[0], arch, 'diverse_init', 'figures', 'accuracy')
        if not os.path.exists(save_here):
            os.makedirs(save_here)
        plt.savefig(os.path.join(save_here, title + ' ' + dataset + ' ' + folders[i].split('_128')[0] + '.png'))
        plt.clf()

def compare_NFR_traditional_to_PCT_all4(arch, dataset, query):
    total = pd.DataFrame([])
    root = os.path.join('output', dataset, arch, 'diverse_init')


    for q in query:
        for i in range(1, 3):
            path1 = os.path.join(root, 'NFR', q + str(i) +'_128')
            path2 = os.path.join(root, 'PCT', q + str(i) +'_128')

            df1 = pd.DataFrame([])
            df2 = pd.DataFrame([])
            for j in range(21):
                data_old1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
                data_new1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
                y_old1 = data_old1['prediction'] == data_old1['GT']
                y_new1 = data_new1['prediction'] != data_new1['GT']
                NFR1 = (y_old1 & y_new1).mean()
                df1.loc[j, 'train type'] = q + ' Traditional'
                df1.loc[j, 'strategy'] = q
                df1.loc[j, 'NFR'] = NFR1
                df1.loc[j, 'Round'] = j
            df1['mov_avg'] = df1['NFR'].rolling(4).mean()
            for j in range(21):
                data_old2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
                data_new2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
                y_old2 = data_old2['prediction'] == data_old2['GT']
                y_new2 = data_new2['prediction'] != data_new2['GT']
                NFR2 = (y_old2 & y_new2).mean()
                df2.loc[j, 'train type'] = q + ' Positive Congruent'
                df2.loc[j, 'strategy'] = q
                df2.loc[j, 'NFR'] = NFR2
                df2.loc[j, 'Round'] = j
            df2['mov_avg'] = df2['NFR'].rolling(4).mean()

            total = pd.concat([total, df1, df2])

    sns.lineplot(x="Round", y="mov_avg", hue='train type', data=total, ci=30)
    if arch == 'resnet_18':
        title = 'Resnet 18'
    elif arch == 'resnet_50':
        title = 'Resnet 50'
    elif arch == 'vgg_16':
        title = 'VGG 16'
    elif arch == 'densenet_121':
        title = 'Densenet 121'

    plt.title(title + ' ' + dataset + ' ' + query[0] + ' vs ' + query[1], fontsize=13)
    plt.ylabel('NFR', fontsize=13)
    plt.xlabel('Round', fontsize=13)

    plt.show()
    # save_here = os.path.join('excel', dataset.split('Patient')[0], arch, 'diverse_init', 'figures', 'NFR')
    # if not os.path.exists(save_here):
    #     os.makedirs(save_here)
    # plt.savefig(os.path.join(save_here, title + ' ' + dataset + ' ' + query + '.png'))
    # plt.clf()

def get_area_between_curves(arch, dataset, query):

    total_df1 = pd.DataFrame([])
    total_df2 = pd.DataFrame([])
    root = os.path.join('output', dataset, arch, 'diverse_init')

    for q in query:
        for i in range(1, 3):
            path1 = os.path.join(root, 'NFR', q + str(i) +'_128')
            path2 = os.path.join(root, 'PCT', q + str(i) +'_128')

            df1 = pd.DataFrame([])
            df2 = pd.DataFrame([])
            for j in range(21):
                data_old1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
                data_new1 = pd.read_excel(os.path.join(path1, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
                y_old1 = data_old1['prediction'] == data_old1['GT']
                y_new1 = data_new1['prediction'] != data_new1['GT']
                NFR1 = (y_old1 & y_new1).mean()
                df1.loc[j, 'train type'] = q + ' Traditional'
                df1.loc[j, 'strategy'] = q
                df1.loc[j, 'NFR'] = NFR1
                df1.loc[j, 'Round'] = j
            df1['mov_avg'] = df1['NFR'].rolling(4).mean()
            for j in range(21):
                data_old2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j) + '.xlsx'), index_col=0, engine='openpyxl')
                data_new2 = pd.read_excel(os.path.join(path2, 'positive_cong_test_info' + str(j + 1) + '.xlsx'), index_col=0, engine='openpyxl')
                y_old2 = data_old2['prediction'] == data_old2['GT']
                y_new2 = data_new2['prediction'] != data_new2['GT']
                NFR2 = (y_old2 & y_new2).mean()
                df2.loc[j, 'train type'] = q + ' Positive Congruent'
                df2.loc[j, 'strategy'] = q
                df2.loc[j, 'NFR'] = NFR2
                df2.loc[j, 'Round'] = j
            df2['mov_avg'] = df2['NFR'].rolling(4).mean()

            total_df1 = pd.concat([total_df1, df1], axis=1)
            total_df2 = pd.concat([total_df2, df2], axis=1)

        print('Area between ' + q + ' Traditional and ' + q + ' Positive Congruent')
        # c = total_df1["mov_avg"][3:]
        # d = total_df2["mov_avg"][3:]
        # a = total_df1["mov_avg"][3:].mean(axis=1).tolist()
        # b = total_df2["mov_avg"][3:].mean(axis=1).tolist()
        compute_area_between_curves(total_df1["mov_avg"][3:].mean(axis=1).tolist(), total_df2["mov_avg"][3:].mean(axis=1).tolist())


def compute_area_between_curves(x_y_curve1, x_y_curve2):
    polygon_points = []  # creates a empty list where we will append the points to create the polygon

    for i, xyvalue in enumerate(x_y_curve1):
        polygon_points.append([i, xyvalue])  # append all xy points for curve 1

    for i, xyvalue in enumerate(x_y_curve2[::-1]):
        polygon_points.append([len(x_y_curve2) - i, xyvalue])  # append all xy points for curve 2 in the reverse order
                                             # (from last point to first point)

    for i, xyvalue in enumerate(x_y_curve1[0:1]): # append first point in curve 1 to first point in curve 2 to "close" polygon
        polygon_points.append([i, xyvalue])  # append the first point in curve 1 again, to it "closes" the polygon

    polygon = Polygon(polygon_points)

    area = polygon.area

    x,y = polygon.exterior.xy
    # plt.plot(x, y)
    #
    # original data
    ls = LineString(np.c_[x, y])
    # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    lr.is_simple  # False
    mls = unary_union(lr)
    mls.geom_type  # MultiLineString'

    Area_cal =[]

    for polygon in polygonize(mls):
        Area_cal.append(polygon.area)
        Area_poly = (np.asarray(Area_cal).sum())
    print(Area_poly)
    # plt.show()


if __name__ == "__main__":

    # -------------           ******************             -------------#
    # alg_names = ['Random', 'Margin', 'Patient Diverse Margin']
    # folders = ['rand_512', 'margin_512', 'patient_diverse_margin_512']
    # #
    # alg_names = ['Random', 'Entropy', 'Patient Diverse Entropy']
    # folders = ['rand_512', 'entropy_512', 'patient_diverse_entropy_512']
    #
    # alg_names = ['Random', 'Least Confidence', 'Patient Diverse Least Confidence']
    # folders = ['rand_512', 'least_conf_512', 'patient_diverse_least_conf_512']
    #
    # alg_names = ['Random', 'Patient Diverse Random']
    # folders = ['rand_512', 'patient_diverse_512']
    # #
    # alg_names = ['Random', 'BADGE', 'Patient Diverse BADGE']
    # folders = ['rand_512', 'badge_512', 'patient_diverse_badge_512']

    # alg_names = ['Random', 'Entropy', 'Clinically Diverse Entropy']
    # folders = ['rand_64', 'entropy_64', 'clinically_diverse_entropy_64']

    # alg_names = ['Random', 'BADGE', 'Clinically Diverse BADGE']
    # folders = ['rand_64', 'badge_64', 'clinically_diverse_badge_64']
    #
    # alg_names = ['Random', 'Clinically Diverse Random']
    # folders = ['rand_64', 'clinically_diverse_64']

    # def generate_plot_journal(alg_names, init,  arch, dataset, folders):
    #
    #     root = os.path.join('excel', dataset, arch, init)
    #     paths = []
    #     for i in range(len(folders)):
    #         paths.append(os.path.join(root, folders[i]) + '/')
    #
    #     plot_lc(path_list=paths, names=alg_names, plotting_col='Test Acc', xlim=(0, 3000), ci=30)
    #
    # generate_plot_journal(alg_names, 'diverse_init', 'resnet_18', 'RCT', folders)
    # -------------           ******************             -------------#

    # vis_variance_per_trial(paths, alg_names)
    # ------- TO COMPARE INITIALIZATION EXPERIMENTS ---------
    # rand_paths = ['excel/Kermany/densenet_121/rand_init/rand_128/', 'excel/Kermany/densenet_121/rand_init/badge_128/',
    #               'excel/Kermany/densenet_121/rand_init/entropy_128/',
    #          'excel/Kermany/densenet_121/rand_init/patient_diverse_128/', 'excel/Kermany/densenet_121/rand_init/patient_diverse_entropy_128/',
    #          'excel/Kermany/densenet_121/rand_init/patient_diverse_entropy_macro_128/', 'excel/Kermany/densenet_121/rand_init/least_conf_128/',
    #          'excel/Kermany/densenet_121/rand_init/margin_128/', 'excel/Kermany/densenet_121/rand_init/patient_diverse_margin_128/',
    #          'excel/Kermany/densenet_121/rand_init/patient_diverse_least_conf_128/', 'excel/Kermany/densenet_121/rand_init/patient_diverse_badge_128/']
    # #
    # diverse_paths = ['excel/Kermany/densenet_121/diverse_init/rand_128/', 'excel/Kermany/densenet_121/diverse_init/badge_128/',
    #                  'excel/Kermany/densenet_121/diverse_init/entropy_128/',
    #          'excel/Kermany/densenet_121/diverse_init/patient_diverse_128/', 'excel/Kermany/densenet_121/diverse_init/patient_diverse_entropy_128/',
    #          'excel/Kermany/densenet_121/diverse_init/patient_diverse_entropy_macro_128/', 'excel/Kermany/densenet_121/diverse_init/least_conf_128/',
    #          'excel/Kermany/densenet_121/diverse_init/margin_128/', 'excel/Kermany/densenet_121/diverse_init/patient_diverse_margin_128/',
    #          'excel/Kermany/densenet_121/diverse_init/patient_diverse_least_conf_128/', 'excel/Kermany/densenet_121/diverse_init/patient_diverse_badge_128/']
    # #
    # alg_names = ['Random', 'BADGE', 'Entropy', 'Patient Diverse Random', 'Patient Diverse Entropy (micro)',
    #              'Patient Diverse Entropy (macro)', 'Least Confidence', 'Margin', 'Patient Diverse Margin',
    #              'Patient Diverse Least Confidence', 'Patient Diverse BADGE']

    # vis_compare_variance_per_trial(rand_paths, diverse_paths, alg_names)
    # vis_compare_variance_per_round(rand_paths, diverse_paths, alg_names, rounds=[2])
    # vis_variance_per_round(paths, alg_names, rounds=[0, 1, 2, 3])
    # paths = ['excel/STL10/rand/', 'excel/STL10/rev_idealEF/', 'excel/STL10/idealEF/']
    # alg_names = ['Random', 'Most Forgotten', 'Least Forgotten']
    # plot_lc(path_list=paths, names=alg_names, plotting_col='Easy Test Acc', xlim=(0, 2500), ylim=(90, 100))
    # plot_lc(path_list=paths, names=alg_names, plotting_col='Time', xlim=(0, 20000))

    # visualize_forgetting_events('./output/rand2/')
    # visualize_example_forgetting_sequence(folder_list=['./output/KermanyPatient/diverse_init/patient_diverse6/'],
    #                                       index_list=[0, 15, 30, 37], prefix='test_')
    # visualize_forgetting_events_confusion_matrix('./output/KermanyPatient/badge1/', prefix='test_')
    #plot_lc(path_list=paths, names=alg_names, plotting_col='Difficult Test Acc', xlim=(0, 16000), ylim=(35, 70))
    #plot_lc(path_list=paths, names=alg_names, plotting_col='Easy Test Acc', xlim=(0, 16000), ylim=(70, 98))
    # plot_lc(path_list=paths, names=alg_names, plotting_col='Test Acc', xlim=(0, 16000), ylim=(35, 80))

    # -------------           ******************             -------------#
    # kermany_patient_dist('test')
    # kermany_xray_patient_dist('train')
    # -------------           ******************             -------------#

    # visualize_patient_accuracy_confusion_matrix('./output/KermanyPatient/diverse_init/patient_diverse5/',
    #                                             name='test_accuracy', prefix='patient_')
    # patient_intersection_train_test()

    # ---------------- ************** ----------------------------
    # standard_folders = ['rand_128', 'badge_128', 'entropy_128', 'least_conf_128', 'margin_128']
    # diverse_folders = ['patient_diverse_128', 'patient_diverse_badge_128', 'patient_diverse_entropy_128',
    #                  'patient_diverse_least_conf_128', 'patient_diverse_margin_128']
    #
    # standard_names = ['Random', 'BADGE', 'Entropy', 'Least Confidence', 'Margin']
    # diverse_names = ['Patient Diverse Random', 'Patient Diverse BADGE', 'Patient Diverse Entropy',
    #                  'Patient Diverse Least Confidence', 'Patient Diverse Margin']
    #
    # def strategy_variance_per_trial_plots(standard_names, diverse_names, arch, dataset, standard_folders,
    #                                       diverse_folders):
    #     root = os.path.join('excel', dataset, arch, 'diverse_init')
    #     std_paths = []
    #     div_paths = []
    #     for i in range(len(standard_folders)):
    #         std_paths.append(os.path.join(root, standard_folders[i]) + '/')
    #         div_paths.append(os.path.join(root, diverse_folders[i]) + '/')
    #
    #     vis_strategy_variance_per_trial(std_paths, div_paths, standard_names, diverse_names)
    #
    #
    # strategy_variance_per_trial_plots(standard_names, diverse_names, 'densenet_121', 'Kermany', standard_folders,
    #                                   diverse_folders)
    # # ---------------- ************** ----------------------------

       # ************* New initialization experiments *****************
    # compare_initializations(arch='vgg_16', dataset='KermanyXrayPatient', init='rand_init', nstart=1000,
    #                         pretrained=False)
    # queries = [['rand', 'patient_diverse'], ['entropy', 'patient_diverse_entropy'],
    #            ['margin', 'patient_diverse_margin'], ['least_conf', 'patient_diverse_least_conf'],
    #            ['badge', 'patient_diverse_badge']]
    queries = [['rand', 'clinically_diverse'], ['entropy', 'clinically_diverse_entropy'],
               ['badge', 'clinically_diverse_badge']]

    for query in queries:
        compare_NFR(arch='resnet_18', dataset='RCTPatient', query=query, metric='PCT')

    # queries = ['rand', 'patient_diverse', 'entropy', 'patient_diverse_entropy',
    #            'margin', 'patient_diverse_margin', 'least_conf', 'patient_diverse_least_conf',
    #            'badge', 'patient_diverse_badge']
    # queries = ['rand', 'badge', 'entropy', 'patient_diverse', 'patient_diverse_badge', 'patient_diverse_entropy']
    # for query in queries:
    #     compare_NFR_traditional_to_PCT(arch='resnet_18', dataset='KermanyPatient', query=query)
    # CIFAR10Traditional
    # KermanyPatient
    # KermanyTraditional
    # compare_accuracy_traditional_to_PCT(arch='vgg_16', dataset='KermanyPatient', xlim=(0, 2500))

    # queries = [['rand', 'patient_diverse'], ['entropy', 'patient_diverse_entropy'],
    #            ['margin', 'patient_diverse_margin'], ['least_conf', 'patient_diverse_least_conf'],
    #            ['badge', 'patient_diverse_badge']]
    # queries = [['rand', 'patient_diverse']]
    # queries = [['rand', 'patient_diverse'], ['entropy', 'patient_diverse_entropy'],
    #            ['badge', 'patient_diverse_badge']]
    # for query in queries:
    #     compare_NFR_traditional_to_PCT_all4(arch='resnet_50', dataset='KermanyPatient', query=query)

    # compare_accuracy_traditional_to_PCT(arch='resnet_18', dataset='CIFAR10Traditional')
    # for query in queries:
    #     get_area_between_curves(arch='densenet_121', dataset='KermanyPatient', query=query)

# if __name__ == "__main__":
#     visualize_forgetting_events('output_resnet18_50epochs/')
