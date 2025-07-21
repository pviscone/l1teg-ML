#%%
import sys
sys.path.append("../utils")
from file_utils import openAsDataframe
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio"}

df = openAsDataframe("/eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root", "TkEle")

#%%

features = [
    'TkEle_idScore', "TkEle_caloEta",
    'TkEle_in_caloLooseTkWP', 'TkEle_in_caloPt', 'TkEle_in_caloRelIso',
    'TkEle_in_caloSS', 'TkEle_in_caloStaWP', 'TkEle_in_caloTkAbsDeta',
    'TkEle_in_caloTkAbsDphi', 'TkEle_in_caloTkNMatch',
    'TkEle_in_caloTkPtRatio', 'TkEle_in_tkChi2RPhi', 'TkEle_in_tkPtFrac'
]

df_train, df_test, gen_train, gen_test, ptratio_train, ptratio_test= train_test_split(df[features], df["TkEle_Gen_pt"], df["TkEle_Gen_ptRatio"], test_size=0.2, random_state=42)
# %%

#Placeholder
#enable batching (max 10k points)
model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(df_train, gen_train)


#%%
from plot_utils import plot_ptratio_distributions, response_plot
#evaluate on test set
df_test["TkEle_Gen_ptRatio"] = ptratio_test
df_test["TkEle_Gen_pt"] = gen_test
df_test["TkEle_regressedPtRatio"] = model.predict(df_test)/df_test["TkEle_Gen_pt"]

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio"}


eta_bins, centers, medians, perc5s, perc95s= plot_ptratio_distributions(
                            df_test,
                            ptratio_dict,
                            genpt_,
                            eta_,)
response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s)
# %%
