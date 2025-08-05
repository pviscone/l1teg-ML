#%%
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

os.makedirs('plots/step3_feature_importance', exist_ok=True)

model = xgb.XGBRegressor()
model.load_model('models/xgboost_model_L1_q10.json')

features = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    #'TkEle_idScore',
    "$| \eta |$",
    r"$| \Delta \phi_{caloTk} |$",
    r"$\chi^2_{tk}$",
    r"$p_T^{Calo}$",
    #'caloRelIso',
    r"$E_{2 \times 5}/E_{5 \times 5}$",
    #'tkPtFrac',
    #'caloTkNMatch',
    r"$p_T^{Calo}/p_T^{Tk}$",
]


importances = model.feature_importances_
plt.figure(figsize=(10, 6))
#argsort importances
sorted_indices = importances.argsort()
features = [features[i] for i in sorted_indices]
importances = importances[sorted_indices]
plt.barh(features, importances)
plt.xlabel("Feature Importance")
hep.cms.text("Phase-2 Simulation Preliminary")
plt.tight_layout()
plt.savefig("plots/step3_feature_importance/feature_importance.png")
plt.savefig("plots/step3_feature_importance/feature_importance.pdf")
plt.show()