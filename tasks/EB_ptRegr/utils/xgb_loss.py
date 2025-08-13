import numpy as np

class L1Loss:
    def __init__(self, w, cls, alpha=1, debug=False):
        self.w = w
        self.cls = cls
        self.alpha = alpha
        self.debug = debug

    def loss(self, y, preds):
        if self.debug:
            print("Starting custom loss calculation")
            print(f"Weights: {self.w}")
            print(f"Classes: {self.cls}")
            print(f'Predictions: {preds}')
            print(f'Targets: {y}')
            print("#####\n")
        diff = (preds - y)
        grad = self.w * (self.cls * np.sign(diff) + (1-self.cls) * self.alpha * (preds > 1))
        hess = self.w * (self.cls + (1-self.cls) * self.alpha)
        return grad, hess

class L1Metrics:
    def __init__(self, w, cls):
        self.w = w
        self.cls = cls


        self.bkg_mae=np.array([])
        self.bkg_median=np.array([])
        self.bkg_quant95=np.array([])
        self.sig_mae=np.array([])
        self.sig_median=np.array([])
        self.sig_quant95=np.array([])
        self.mae=np.array([])

    def sig_metric(self, y, preds):
        self.sig_mae = np.append(self.sig_mae,np.mean(self.cls*np.abs(preds -y)))
        self.sig_quant95 = np.append(self.sig_quant95,np.quantile(preds[self.cls==1],0.95))
        self.sig_median = np.append(self.sig_median, np.median(preds[self.cls==1]))

    def bkg_metric(self, y, preds):
        self.bkg_quant95 = np.append(self.bkg_quant95,np.quantile(preds[self.cls==0],0.95))
        self.bkg_mae=np.append(self.bkg_mae,np.mean(np.abs(preds[self.cls==0])))
        self.bkg_median = np.append(self.bkg_median, np.median(preds[self.cls==0]))

    def mae_metric(self, y, preds):
        return np.mean(np.abs(preds-y))

    def metrics(self, y, preds):
        self.sig_metric(y, preds)
        self.bkg_metric(y, preds)
        self.mae = np.append(self.mae,self.mae_metric(y, preds))
        return self.sig_mae[-1]