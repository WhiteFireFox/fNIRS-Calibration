import numpy as np
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import random
import os
import metrics

# Select dataset through task_id
task = ['UFFT', 'MA']
task_id = 0
print(task[task_id])

if task_id == 0:
    # UFFT
    all_sub = 30
elif task_id == 1:
    # MA
    all_sub = 29

all_predicts = np.array([])
all_labels = np.array([])

init_flag = False

for n_sub in range(1, all_sub+1):
    for tr in range(5):
        path = os.path.join('save', task[task_id], 'KFold', str(n_sub), str(tr+1))
        predicts = np.load(os.path.join(path, "outputs.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
        if not init_flag:
            all_predicts = np.squeeze(predicts)
            all_labels = np.squeeze(labels)
            init_flag = True
        else:
            all_predicts = np.concatenate([all_predicts, np.squeeze(predicts)])
            all_labels = np.concatenate([all_labels, np.squeeze(labels)])

all_predicts_softmax = np.exp(all_predicts) / np.reshape(np.sum(np.exp(all_predicts), axis=1), [-1, 1])

all_labels = all_labels.astype(np.int64)

y_pred = np.max(all_predicts_softmax, axis=1)

y_true = np.equal(np.argmax(all_predicts_softmax, axis=1), all_labels).astype(np.int32)

print("Confidence: " + str(np.average(y_pred)))

print("Accuracy: " + str(np.average(y_true)))

ece_criterion = metrics.ECELoss()
print('ECE: %f' % (ece_criterion.loss(all_predicts, all_labels, n_bins = 10)))

mce_criterion = metrics.MCELoss()
print('MCE: %f' % (mce_criterion.loss(all_predicts, all_labels, n_bins = 10)))

oe_criterion = metrics.OELoss()
print('OE: %f' % (oe_criterion.loss(all_predicts, all_labels, n_bins = 10)))

sce_criterion = metrics.SCELoss()
print('SCE: %f' % (sce_criterion.loss(all_predicts, all_labels, n_bins = 10)))

ace_criterion = metrics.ACELoss()
print('ACE: %f' % (ace_criterion.loss(all_predicts, all_labels, n_bins = 10)))

tace_criterion = metrics.TACELoss()
threshold = 0.01
print('TACE (threshold = %f): %f' % (threshold, tace_criterion.loss(all_predicts, all_labels, threshold, n_bins = 10)))

prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

disp = CalibrationDisplay(prob_true, prob_pred, y_pred)

disp.plot()
plt.show()