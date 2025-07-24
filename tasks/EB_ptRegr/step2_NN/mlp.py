#%%
import sys
import os
sys.path.append("../utils")
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from file_utils import openAsDataframe
from compute_weights import cut_and_compute_weights
from keras.models import Sequential
from keras.layers import Dense
import jax
import keras

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio"}

metric = "L1"

if metric == "L1":
    loss = "mae"
    w = "wTot"
elif metric == "L2":
    loss = "mse"
    w = "w2Tot"
else:
    raise ValueError("Unknown metric. Use 'L1' or 'L2'.")

if not os.path.exists("DoubleElectron_PU200.root"):
    raise ValueError("xrdcp root://eosuser.cern.ch//eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root .")

df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")
df = cut_and_compute_weights(df, genpt_, pt_)


features = [
    "TkEle_caloEta",
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    'TkEle_in_caloTkAbsDphi',
    'TkEle_in_tkChi2RPhi',
    #'TkEle_in_caloLooseTkWP',
    'TkEle_in_caloPt',
    'TkEle_in_caloRelIso',
    'TkEle_in_caloSS',
    'TkEle_in_tkPtFrac',
    'TkEle_in_caloTkNMatch',
    'TkEle_in_caloTkPtRatio',
    #'TkEle_idScore',
]

df_train, df_test, gen_train, gen_test, ptratio_train, ptratio_test, eta_train, eta_test, dfw_train, dfw_test = train_test_split(df[features], df["TkEle_Gen_pt"], df["TkEle_Gen_ptRatio"], df[eta_], df[["RESw", "BALw", "wTot","w2Tot"]], test_size=0.2, random_state=42)
# %%


print("JAX devices:", jax.devices())

model = Sequential([
    Dense(20, activation='relu', input_shape=(len(features),)),
    Dense(20, activation='relu'),
    Dense(20, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1)
])

initial_learning_rate = 0.0005
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss=loss, jit_compile=True)


# Plot training & validation loss values
history = model.fit(
    df_train, gen_train,
    sample_weight=dfw_train[w].values,
    epochs=50,
    batch_size=256,
    validation_data=(df_test, gen_test),
    verbose=1
)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
#%%
from plot_utils import plot_ptratio_distributions, response_plot  # noqa: E402
#evaluate on test set
def plot_results(model, plot_distributions=False):
    global ptratio_test, ptratio_dict, gen_test, genpt_, eta_test, eta_
    df_test[ptratio_dict["NoRegression"]] = ptratio_test
    df_test[genpt_] = gen_test
    df_test[eta_]=eta_test
    df_test[ptratio_dict["Regressed"]] = model.predict(df_test[features].values)[:,0]/df_test[genpt_]



    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances = plot_ptratio_distributions(df_test,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(4,100,33), plots=plot_distributions, savefolder=f"plots{metric}")
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=f"plots{metric}")

plot_results(model, plot_distributions=True)
# %%

