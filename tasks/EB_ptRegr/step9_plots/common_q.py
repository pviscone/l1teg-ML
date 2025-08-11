import numpy as np

features_q = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    #'TkEle_idScore',
    "hwCaloEta",
    'caloTkAbsDphi',
    'hwTkChi2RPhi',
    'caloPt',
    #'caloRelIso',
    'caloSS',
    #'tkPtFrac',
    #'caloTkNMatch',
    'caloTkPtRatio',
]

def scale(df):
    df["hwCaloEta"] = np.clip(-1+df["TkEle_hwCaloEta"].abs()/2**8, -1, 1)
    df["caloTkAbsDphi"] = np.clip(-1 + df["TkEle_in_caloTkAbsDphi"]/2**5, -1, 1)
    df["hwTkChi2RPhi"] = np.clip(-1 + df["TkEle_in_hwTkChi2RPhi"]/2**3, -1, 1)
    df["caloPt"] = np.clip(-1+(df["TkEle_in_caloPt"]-1)/2**6, -1, 1)
    df["caloRelIso"] = np.clip(-1 + df["TkEle_in_caloRelIso"]*2**3, -1, 1)
    df["caloSS"] = np.clip(-1 + df["TkEle_in_caloSS"]*2, -1, 1)
    df["tkPtFrac"] = np.clip(-1 + df["TkEle_in_tkPtFrac"]/2**3, -1, 1)
    df["caloTkNMatch"] = np.clip(-1 + df["TkEle_in_caloTkNMatch"]/2**2, -1, 1)
    df["caloTkPtRatio"] = np.clip(-1 + df["TkEle_in_caloTkPtRatio"]/2**3, -1, 1)
    return df

init_pred = 512 * 2**-9