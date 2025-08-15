#%%
import sys
import os
sys.path.append("..")
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import jax
import keras


from common import signal_train, pt_, w, features_q, ptratio_dict, genpt_, eta_
from plot_utils import plot_results
from file_utils import open_signal

metric = "L1"
if metric == "L1":
    loss = "mae"

#?Open dfs
df = open_signal(signal_train)

#? Split into train and test and mask the train
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

train_mask = np.bitwise_and( df_train["target"] < 2 , df_train[pt_].values >=4)
#train_mask = (df_train["target"] < out_cut- 1./2**(q_out[0]-q_out[1]))
df_train = df_train[train_mask]

# %%


print("JAX devices:", jax.devices())

model = Sequential([
    Dense(50, activation='relu', input_shape=(len(features_q),)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1)
])

initial_learning_rate = 0.0005
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss=loss, jit_compile=True)


# Plot training & validation loss values
history = model.fit(
    df_train[features_q], df_train["target"].values,
    sample_weight=df_train[w].values,
    epochs=20,
    batch_size=256,
    validation_data=(df_test[features_q], df_test["target"].values),
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
df_test[ptratio_dict["Regressed"]] = model.predict(df_test[features_q])[:,0] * df_test[pt_].values/ df_test[genpt_].values
#%%
plot_results(df_test, ptratio_dict, genpt_, eta_, verbose=False, savefolder=f"plots{metric}/NN")
plot_results(df_test, ptratio_dict, genpt_, eta_, verbose=False, savefolder=f"plots{metric}/NN", eta_bins=np.array([0,1.479]))
# %%
#save model
#model.save(f"../models/NN_{metric}_model.keras")