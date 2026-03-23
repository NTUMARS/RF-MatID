from metrics.accumulator import Accumulator
# from accumulator import Accumulator
import torch

class BC_Metrics:
    def __init__(self, threshold: float = 0.5, defect_class_idx: int = 1):
        self.threshold = threshold
        self.accumulator = Accumulator()
        self.defect_class_idx = defect_class_idx
    
    def evaluate(self, pred: torch.Tensor, gt: torch.Tensor):
        '''
        pred : torch.Tensor : shape (B, 2), contain the prediction score for each class in floating format
        gt : torch.Tensor : shape (B, 1), contain the ground truth class index
        
        We assume that the class index is 1 for the class of defect and 0 for the class of non-defect,
        Thus the ground truth tensor should contain 1 for defect and 0 for non-defect,
        and the true positive is pred_highest_idx:gt = 1:1, false positive is pred_highest_idx:gt = 1:0,
        false negative is pred_highest_idx:gt = 0:1, and true negative is pred_highest_idx:gt = 0:0
        '''
        # Check if the prediction is normalized by SoftMax
        if torch.sum(pred) == len(pred):
            normalized_pred = pred
        else:
            normalized_pred = torch.softmax(pred, dim=1)

        pred_mask = normalized_pred >= self.threshold
        predicted_label = pred_mask[:,self.defect_class_idx]
        gt_mask = gt == self.defect_class_idx
        tp = torch.logical_and(predicted_label, gt_mask).sum()
        fp = torch.logical_and(predicted_label, ~gt_mask).sum()
        fn = torch.logical_and(~predicted_label, gt_mask).sum()
        tn = torch.logical_and(~predicted_label, ~gt_mask).sum()
        
        self.accumulator.inc_TP_prediction(tp)
        self.accumulator.inc_FP_prediction(fp)
        self.accumulator.inc_FN_prediction(fn)
        self.accumulator.inc_TN_prediction(tn)
        
    def precision(self):
        total_predicted = self.accumulator.TP + self.accumulator.FP
        if total_predicted == 0:
            total_gt = self.accumulator.TP + self.accumulator.FN
            if total_gt == 0:
                return 1.
            else:
                return 0.
        return float(self.accumulator.TP) / total_predicted
    
    def recall(self):
        total_gt = self.accumulator.TP + self.accumulator.FN
        if total_gt == 0:
            return 1.
        return float(self.accumulator.TP) / total_gt
    
    def accuracy(self):
        total = self.accumulator.TP + self.accumulator.FP + self.accumulator.FN + self.accumulator.TN
        if total == 0:
            return 1.
        return float(self.accumulator.TP + self.accumulator.TN) / total
    
    def num_tp(self):
        return self.accumulator.TP
    
    def num_fp(self):
        return self.accumulator.FP
    
    def num_fn(self):
        return self.accumulator.FN
    
    def num_tn(self):
        return self.accumulator.TN
    
    def F1_score(self):
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0.
        return 2 * precision * recall / (precision + recall)
    
    def __str__(self):
        str = ""
        str += "True positives : {}\n".format(self.num_tp())
        str += "False positives : {}\n".format(self.num_fp())
        str += "False Negatives : {}\n".format(self.num_fn())
        str += "True Negatives : {}\n".format(self.num_tn())
        str += "Precision : {}\n".format(self.precision())
        str += "Recall : {}\n".format(self.recall())
        str += "Accuracy : {}\n".format(self.accuracy())
        str += "F1 Score : {}\n".format(self.F1_score())
        return str
        

class MCC_Metrics:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.classes_accumulators = []
        self.init_accumulators()
        
    def init_accumulators(self):
        for _ in range(self.n_classes):
            self.classes_accumulators.append(Accumulator())
    
    def evaluate(self, pred: torch.Tensor, gt: torch.Tensor):
        """compute the confusion matrix for each class

        Args:
            pred : torch.Tensor : shape (B, N), contain the prediction score for each class in floating format
            gt (torch.Tensor): shape (B, 1), contain the ground truth class index
        """
        
        if torch.sum(pred) == len(pred):
            normalized_pred = pred
        else:
            normalized_pred = torch.softmax(pred, dim=1)
            
        for i, acc in enumerate(self.classes_accumulators):
            pred_mask = normalized_pred.argmax(dim=1) == i
            gt_mask = gt == i
            tp = torch.logical_and(pred_mask, gt_mask).sum()
            tn = torch.logical_and(~pred_mask, ~gt_mask).sum()
            fp = torch.logical_and(pred_mask, ~gt_mask).sum()
            fn = torch.logical_and(~pred_mask, gt_mask).sum()
            acc.inc_TP_prediction(tp)
            acc.inc_FP_prediction(fp)
            acc.inc_FN_prediction(fn)
            acc.inc_TN_prediction(tn)
            
    def micro_precision(self):
        total_tp = sum([acc.TP for acc in self.classes_accumulators])
        total_fp = sum([acc.FP for acc in self.classes_accumulators])
        if total_tp + total_fp == 0:
            return 1.
        return total_tp / (total_tp + total_fp)
    
    def classes_precisions(self):
        precision_list = []
        for acc in self.classes_accumulators:
            if acc.TP + acc.FP == 0:
                precision_list.append(1.)
            else:
                precision_list.append(acc.TP / (acc.TP + acc.FP))
        return precision_list
    
    def macro_precision(self):
        total_precision = 0
        precision_list = self.classes_precisions()
        for temp_precision in precision_list:
            total_precision += temp_precision
        return total_precision / self.n_classes
    
    def micro_recall(self):
        total_tp = sum([acc.TP for acc in self.classes_accumulators])
        total_fn = sum([acc.FN for acc in self.classes_accumulators])
        if total_tp + total_fn == 0:
            return 1.
        return total_tp / (total_tp + total_fn)
    
    def classes_recalls(self):
        recall_list = []
        for acc in self.classes_accumulators:
            if acc.TP + acc.FN == 0:
                recall_list.append(1.)
            else:
                recall_list.append(acc.TP / (acc.TP + acc.FN))
        return recall_list
    
    def macro_recall(self):
        total_recall = 0
        recall_list = self.classes_recalls()
        for temp_recall in recall_list:
            total_recall += temp_recall
        return total_recall / self.n_classes
    
    def micro_f1_score(self):
        precision = self.micro_precision()
        recall = self.micro_recall()
        if precision + recall == 0:
            return 0.
        return 2 * precision * recall / (precision + recall)
    
    def macro_f1_score(self):
        total_f1 = 0
        precision_list = self.classes_precisions()
        recall_list = self.classes_recalls()
        for i in range(self.n_classes):
            temp_precision = precision_list[i]
            temp_recall = recall_list[i]
            temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall)
            if temp_precision + temp_recall == 0:
                temp_f1 = 0.
            total_f1 += temp_f1
        return total_f1 / self.n_classes
    
    def accuracy(self):
        """
        Number of correctly predicted instances divided by the total number of instances
        """
        total_instance = 0
        for acc in self.classes_accumulators:
            total_instance += acc.TP + acc.FP + acc.FN + acc.TN
            break
        if total_instance == 0:
            raise ValueError("No instance to evaluate, total_instance == 0")
        total_tp = sum([acc.TP for acc in self.classes_accumulators])
        return total_tp / total_instance

    def num_tp(self):
        tp_list = [float(acc.TP) for acc in self.classes_accumulators]
        return tp_list
    
    def num_fp(self):
        fp_list = [float(acc.FP) for acc in self.classes_accumulators]
        return fp_list
    
    def num_fn(self):
        fn_list = [float(acc.FN) for acc in self.classes_accumulators]
        return fn_list
    
    def num_tn(self):
        tn_list = [float(acc.TN) for acc in self.classes_accumulators]
        return tn_list
    
    def __str__(self):
        str = ""
        str += "True positives : {}\n".format(self.num_tp())
        str += "False positives : {}\n".format(self.num_fp())
        str += "False Negatives : {}\n".format(self.num_fn())
        str += "True Negatives : {}\n".format(self.num_tn())
        str += "Precision : micro: {}; macro: {}\n".format(self.micro_precision(), self.macro_precision())
        str += "Recall : micro: {}; macro: {}\n".format(self.micro_recall(), self.macro_recall())
        str +="F1 Score : micro: {}; macro: {}\n".format(self.micro_f1_score(), self.macro_f1_score())
        str += "Accuracy : {}\n".format(self.accuracy())
        return str