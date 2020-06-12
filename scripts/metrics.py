'''
Authors: Geeticka Chauhan, Ruizhi Liao
This script contains metrics for evaluating model predictions
'''
from scipy.stats import logistic
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import sklearn
from scipy.special import softmax

'''
Evaluation related helpers
'''
def convert_ordinal_label_to_labels(ordinal_label):
    return np.sum(ordinal_label)
    # in the case of gold labels, convert ordinal labels to predictions

def convert_sigmoid_prob_to_labels(pred):
    sigmoid_pred = logistic.cdf(pred)
    threshold = 0.5
    if sigmoid_pred[0] > threshold:
        if sigmoid_pred[1] > threshold:
            if sigmoid_pred[2] > threshold:
                return 3
            else:
                    return 2
        else:
            return 1
    else:
        return 0

def compute_ordinal_auc_from_multiclass(labels_raw, preds):
    """
    Given the 4 channel output of multiclass, compute the 3 channel ordinal auc
    """

    if len(labels_raw) != len(preds):
        raise ValueError('The size of the labels does not match the size the preds!')
    num_datapoints = len(labels_raw) # labels_raw needs to be between 0 and 1
    if len(preds[0]) != 4:
        raise ValueError('This auc can only be computed for multiclass')
    desired_channels = 3
    ordinal_aucs = [] # 0v123, 01v23, 012v3
    for i in range(desired_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(min(1.0, max(0.0, labels_raw[j] - i))) # if gold is 3 and channel is 0v123, then y is 1
            pred.append(sum(preds[j][i+1 : desired_channels+1])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        ordinal_auc_val = round(sklearn.metrics.auc(fpr, tpr), 4)
        ordinal_aucs.append(ordinal_auc_val)
    return ordinal_aucs

def compute_ordinal_auc_onehot_encoded(labels, preds):
    """
    Given the 4 channel output of multiclass, compute the 3 channel ordinal auc
    """

    if len(labels) != len(preds):
        raise ValueError('The size of the labels does not match the size the preds!')
    num_datapoints = len(labels)
    if len(preds[0]) != 4:
        raise ValueError('This auc can only be computed for multiclass')
    desired_channels = 3
    ordinal_aucs = [] # 0v123, 01v23, 012v3
    for i in range(desired_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(sum(labels[j][i+1 : desired_channels+1])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
            pred.append(sum(preds[j][i+1 : desired_channels+1])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        ordinal_auc_val = round(sklearn.metrics.auc(fpr, tpr), 4)
        ordinal_aucs.append(ordinal_auc_val)
    return ordinal_aucs

def compute_auc(labels, preds, output_channel_encoding='multilabel'):
    """
    expects to take labels and preds dimensionality batch/total_data_len X channels
    """

    if len(labels) != len(preds):
        raise ValueError('The size of the labels does not match the size the preds!')
    if output_channel_encoding != 'multilabel' and output_channel_encoding != 'multiclass':
        raise Exception("You can only compute AUC for the multiclass or multilabel case")

    num_datapoints = len(labels)
    num_channels = len(labels[0])
    e_y = np.zeros(num_datapoints) # Label y as an integer in {0,1,2,3}
    e_pred = np.zeros(num_datapoints) # Probabilistic expection of the prediction
    aucs = []
    for i in range(num_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(labels[j][i])
            pred.append(preds[j][i])
            if output_channel_encoding == 'multilabel':
                e_y[j] += labels[j][i]
                e_pred[j] += preds[j][i]
            else:
                # for the one-hot case, E[Y] = 0 * Prob(y=0) + 1 * Prob(y=1) + 2 * Prob(y=2) + 3 * Prob(y=3)
                # E[Y] = 0*channel0 + 1*channel1 + 2*channel2 + 3*channel3
                e_y[j] += i * labels[j][i] 
                e_pred[j] += i * preds[j][i]

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        auc_val = round(sklearn.metrics.auc(fpr, tpr), 4)
        aucs.append(auc_val)

    pairwise_aucs = {}

    if output_channel_encoding == 'multilabel':
        def compute_pairwise_auc(all_y, all_pred, label0, label1):
            y = []
            pred = []
            for j in range(len(all_y)): # index j to indicate that this is num_datapoints dimension
                if all_y[j] == label0 or all_y[j] == label1:
                    y.append(e_y[j])
                    pred.append(all_pred[j])
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=label1)
            return round(sklearn.metrics.auc(fpr, tpr), 4)

        pairwise_aucs['0v1'] = compute_pairwise_auc(e_y, e_pred, 0, 1)
        pairwise_aucs['0v2'] = compute_pairwise_auc(e_y, e_pred, 0, 2)
        pairwise_aucs['0v3'] = compute_pairwise_auc(e_y, e_pred, 0, 3)
        pairwise_aucs['1v2'] = compute_pairwise_auc(e_y, e_pred, 1, 2)
        pairwise_aucs['1v3'] = compute_pairwise_auc(e_y, e_pred, 1, 3)
        pairwise_aucs['2v3'] = compute_pairwise_auc(e_y, e_pred, 2, 3)

    '''
    Another way to compute pairwise auc to say all_y should be one-hot, all_pred should be softmax logits
    Basically changing this into a binary AUC computation by computing the probability of the positive class
    that is arbitrarily chosen as channel 1. We take channel 0 into account when we compute the predicted
    probability using that value
    note you can use the above method for multiclass and use the e_y and e_pred computed for multiclass
    but it biases us by considering all classes in AUC computation rather than just the current class
    '''
    if output_channel_encoding == 'multiclass':
        def compute_pairwise_auc(all_y, all_pred, channel0, channel1):
            y = []
            pred = []
            for j in range(len(all_y)): # j to indicate number of data points dimension
                if all_y[j][channel0] == 1 or all_y[j][channel1] == 1:
                    y.append(all_y[j][channel1])
                    pred.append(all_pred[j][channel1]/(all_pred[j][channel0]+all_pred[j][channel1]))
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
            return round(sklearn.metrics.auc(fpr, tpr), 4)
        pairwise_aucs['0v1'] = compute_pairwise_auc(labels, preds, 0, 1)
        pairwise_aucs['0v2'] = compute_pairwise_auc(labels, preds, 0, 2)
        pairwise_aucs['0v3'] = compute_pairwise_auc(labels, preds, 0, 3)
        pairwise_aucs['1v2'] = compute_pairwise_auc(labels, preds, 1, 2)
        pairwise_aucs['1v3'] = compute_pairwise_auc(labels, preds, 1, 3)
        pairwise_aucs['2v3'] = compute_pairwise_auc(labels, preds, 2, 3)

    return aucs, pairwise_aucs

def get_acc_f1(labels, preds_logits, output_channel_encoding):
    """
    expects dimensionality batch/size_of_data X channels
    expects 3 channel output and converts ordinal values to non ordinal using sigmoid and 
    threshold of 0.5
    """

    # mcc = matthews_corrcoef(labels, preds)
    # in the above case we have to set the threshold
    if output_channel_encoding == 'multilabel':
        new_labels = [convert_ordinal_label_to_labels(ordinal_label) for ordinal_label in labels]
        preds = [convert_sigmoid_prob_to_labels(pred) for pred in preds_logits]
    elif output_channel_encoding == 'multiclass':
        new_labels = labels # assume that labels is passing in int version of 0-3 severity
        preds = np.argmax(preds_logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(new_labels, preds)
    accuracy = accuracy_score(new_labels, preds)
    macro_f1 = np.mean(f1)
    # tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    def round_nd_array(array):
        return [round(val, 4) for val in array]

    return {"accuracy": round(accuracy, 4),
            "f1": round_nd_array(f1),
            "precision": round_nd_array(precision),
            "recall": round_nd_array(recall),
            "macro_f1": round(macro_f1, 4) 
            # "tp": tp,
            # "tn": tn,
            # "fp": fp,
            # "fn": fn
            }, new_labels, preds

def compute_acc_f1_metrics(labels, preds, output_channel_encoding):
    assert len(preds) == len(labels)
    return get_acc_f1(labels, preds, output_channel_encoding) 

def compute_mse(preds_logits, out_label, output_channel_encoding='multilabel'):
    """
    computes the MSE of the 3 channel output using 
    preds = expected value of predictions = sum of the 3 channel outputs 
    squashed by sigmoid (scalar)   
    out_labels = sum of the channel outputs (scalar)
    """
    if output_channel_encoding != 'multilabel' and output_channel_encoding != 'multiclass':
        raise Exception("You can only compute MSE for multiclass or multilabel classification")
    if output_channel_encoding == 'multilabel':
        expectations_predictions = np.sum(logistic.cdf(preds_logits), axis=1) # assuming preds is samples X 3
        out_labels_e = np.sum(out_label, axis=1)
    else:
        preds_probs = softmax(preds_logits, axis=1)
        num_datapoints = len(out_label)
        num_channels = len(out_label[0])
        expectations_predictions = np.zeros(num_datapoints) # Label y as an integer in {0,1,2,3}
        out_labels_e = np.zeros(num_datapoints) # Probabilistic expection of the prediction
        for i in range(num_channels):
            for j in range(num_datapoints):
                # for the one-hot case, E[Y] = 0 * Prob(y=0) + 1 * Prob(y=1) + 2 * Prob(y=2) + 3 * Prob(y=3)
                # E[Y] = 0*channel0 + 1*channel1 + 2*channel2 + 3*channel3
                out_labels_e[j] += i * out_label[j][i] 
                expectations_predictions[j] += i * preds_probs[j][i]
    return round(mse(out_labels_e, expectations_predictions), 4)
