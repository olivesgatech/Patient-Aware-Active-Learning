from os.path import join
import os

im_root = '/home/kaushik/Dropbox (GhassanGT)/OCT/BIGandDATA/ZhangData/chest_xray/'
savePath = '/home/kaushik/Dropbox (GhassanGT)/YashYee/InSync/Big and DATA/OCT/OCT_ActiveLearning/Data/Kermany_xray/'
labels = [('NORMAL', 0), ('PNEUMONIA', 1)]


def createSplit(split):
    total_samples = [('NORMAL', 1349), ('PNEUMONIA', 3883)]
    i = 0
    j = 0
    # Creates text files withe the names of all images in train
    file_object = open(join(savePath, split + '.txt'), 'w')
    for (label, gt) in labels:
        a = os.listdir(join(im_root, split, label))
        for file in os.listdir(join(im_root, split, label)):
            if i == total_samples[j][1]:
                i = 0
                j += 1
            if label == 'PNEUMONIA' and file.startswith('BACTERIA'):  # label == 1
                file_object.write('\n'.join(['PNEUMONIA-' + file + ',' + str(gt) + '\n']))
            elif label == 'PNEUMONIA' and file.startswith('VIRUS'):  # label == 2
                file_object.write('\n'.join(['PNEUMONIA-' + file + ',' + str(gt + 1) + '\n']))
            else:  # label == 0
                file_object.write('\n'.join(['NORMAL-' + file + ',' + str(gt) + '\n']))
            i += 1
    file_object.close()


if __name__ == '__main__':
    createSplit('train')
    createSplit('test')



