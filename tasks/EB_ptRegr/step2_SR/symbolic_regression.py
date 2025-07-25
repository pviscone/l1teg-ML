# %%
import sys
import os

sys.path.append("../utils")

import sympy as sp
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

from file_utils import openAsDataframe
from compute_weights import cut_and_compute_weights

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {
    "NoRegression": "TkEle_Gen_ptRatio",
    "Regressed": "TkEle_regressedPtRatio",
}

metric = "L1"

if metric == "L1":
    loss = "myloss(x, y, w) = w * abs(x - y)"
    w = "wTot"
elif metric == "L2":
    loss = "myloss(x, y, w) = w * (x - y)^2"
    w = "w2Tot"
else:
    raise ValueError("Unknown metric. Use 'L1' or 'L2'.")

if not os.path.exists("DoubleElectron_PU200.root"):
    raise ValueError(
        "xrdcp root://eosuser.cern.ch//eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root ."
    )

df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")
df = cut_and_compute_weights(df, genpt_, pt_)


features = [
    # "TkEle_caloEta",
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloTkAbsDphi',
    #'TkEle_in_tkChi2RPhi',
    #'TkEle_in_caloLooseTkWP',
    "TkEle_in_caloPt",
    #'TkEle_in_caloRelIso',
    #'TkEle_in_caloSS',
    "TkEle_in_tkPtFrac",
    #'TkEle_in_caloTkNMatch',
    "TkEle_in_caloTkPtRatio",
    "TkEle_idScore",
]

(
    df_train,
    df_test,
    gen_train,
    gen_test,
    ptratio_train,
    ptratio_test,
    eta_train,
    eta_test,
    dfw_train,
    dfw_test,
) = train_test_split(
    df[features],
    df["TkEle_Gen_pt"],
    df["TkEle_Gen_ptRatio"],
    df[eta_],
    df[["RESw", "BALw", "wTot", "w2Tot"]],
    test_size=0.2,
    random_state=42,
)
# %%

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "-", "/", "^", ">="],
    unary_operators=[
        "exp",
        # "inv(x) = 1/x",
        "log1p",
        # "logabs(x) = log(abs(x))",
        # "sigm(x) = 1/(1+exp(-x))",
        # "relu",
        # "step(x) = (x > zero(x)) * one(x)"
    ],
    constraints={"pow": (9, 1), "^": (9, 1)},
    extra_sympy_mappings={  # "inv": lambda x: 1 / x,
        # "logabs": lambda x: sp.log(sp.Abs(x)),
        # "sigm": lambda x: 1 / (1 + sp.exp(-x)),
        # "step": lambda x: sp.Piecewise((1, x > 0), (0, True)),
    },
    elementwise_loss=loss,
    batching=True,
    batch_size=8000,
    procs=20,
    populations=20 * 3,
    annealing=True,
    # fast_cycle = True,
    dimensional_constraint_penalty=10**5,
    dimensionless_constants_only=True,
    # ncycles_per_iteration = 5000,
    maxsize=35,
    # precision=16, #Gives errors
    turbo=True,  # enable SIMD
    weight_optimize=0.001,
    model_selection="accuracy",
)

units = ["" for _ in range(len(features))]
units[0] = "kg"
# Train on the SF
model.fit(
    df_train,
    gen_train,
    X_units=units,
    y_units="kg",
    weights=dfw_train[w],
)


# %%
from plot_utils import plot_ptratio_distributions, response_plot  # noqa: E402


# evaluate on test set
def plot_results(model, plot_distributions=False):
    global ptratio_test, ptratio_dict, gen_test, genpt_, eta_test, eta_
    df_test[ptratio_dict["NoRegression"]] = ptratio_test
    df_test[genpt_] = gen_test
    df_test[eta_] = eta_test
    df_test[ptratio_dict["Regressed"]] = model.predict(df_test) / df_test[genpt_]

    (
        eta_bins,
        centers,
        medians,
        perc5s,
        perc95s,
        perc16s,
        perc84s,
        residuals,
        variances,
    ) = plot_ptratio_distributions(
        df_test,
        ptratio_dict,
        genpt_,
        eta_,
        genpt_bins=np.linspace(4, 100, 33),
        plots=plot_distributions,
    )
    response_plot(
        ptratio_dict,
        eta_bins,
        centers,
        medians,
        perc5s,
        perc95s,
        perc16s,
        perc84s,
        residuals,
        variances,
    )


plot_results(model)
# %%
