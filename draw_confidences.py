import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
opt = parser.parse_args()

def _read_confidences(phase, dir_path, dir_name):
    return np.loadtxt(os.path.join(dir_path, dir_name, str(phase) + '-confidences.txt'))

if __name__ == '__main__':
    phases = ['train', 'test', 'rotate90', 'rotate180', 'unknown']

    for phase in phases:
        base_conf = _read_confidences(phase, opt.dir, 'basenet')
        temp_conf = _read_confidences(phase, opt.dir, 'temp_scaling')
        dropout_conf = _read_confidences(phase, opt.dir, 'bayesian_dropout')
        ensemble_conf = _read_confidences(phase, opt.dir, 'deep_ensemble')
        pca_conf = _read_confidences(phase, opt.dir, 'pcanet')

        colors = ['cornflowerblue', 'springgreen', 'wheat', 'tomato', 'violet']
        labels = ['base', 'temp', 'dropout', 'ensemble', 'pca']
        ys, xs, patches = plt.hist([base_conf, temp_conf, dropout_conf, ensemble_conf, pca_conf], bins=10,
                                   range=(0.0, 1.0),
                                   density=False, edgecolor='black', rwidth=0.9, color=colors, label=labels)

        plt.legend()
        plt.xlabel('Confidence(softmax output)')
        plt.ylabel('Num. of samples')
        plt.title(str(phase))

        plt.savefig(os.path.join(opt.dir, str(phase) + '.png'))
        plt.show()