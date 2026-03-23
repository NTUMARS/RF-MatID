class Accumulator:
    def __init__(self):
        self.TP, self.FP, self.FN, self.TN = 0., 0., 0., 0.

    '''
    True positive is the number of correctly predicted class A images
    False positive is the number of wrongly predicted class A images (Non A images predicted as A)
    False negative is the number of wrongly predicted Non-A images (A images predicted as Non-A)
    True negative is the number of rest images
    '''    
    def inc_TP_prediction(self, value=1):
        self.TP += value

    def inc_FN_prediction(self, value=1):
        self.FN += value

    def inc_FP_prediction(self, value=1):
        self.FP += value
        
    def inc_TN_prediction(self, value=1):
        self.TN += value