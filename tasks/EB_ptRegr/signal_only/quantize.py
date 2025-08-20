#%%
import sys
sys.path.append("..")
from common import xgbmodel_signalonly, cpp_cfg, conifer_signalonly, features_q, quant, signal_test
import conifer
from file_utils import open_signal, quantize_features
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgbmodel_signalonly)
hls_model = conifer.converters.convert_from_xgboost(xgb_model, cpp_cfg)
hls_model.compile()
hls_model.save(conifer_signalonly)

#%%
df = open_signal(signal_test)
df = quantize_features(df, features_q, quant)
xgb_full=xgb_model.predict(df[features_q])
cpp_out = hls_model.decision_function(df[features_q].values)[:,0]


diff = xgb_full - cpp_out
print(f"Max diff: {np.max(diff)}")
print(f"Min diff: {np.min(diff)}")
print(f"Mean diff: {np.mean(diff)}")
print(f"Median diff: {np.median(diff)}")


plt.hist(diff,bins=np.linspace(0.95,1.1,100))
plt.axvline(513*2**-9)
plt.savefig("diff_hist.png")