from cmgrdf_cli.data import cms10

def all_processes():
    return {
    "MinBias_train" : {
        "groups" : [
            {
            "name": "MinBias_train",
            "samples": {
                "NuGunAllEta_PU200_train":
                    {
                    "xsec": 31038.96, #(2760.0 * 11246 / 1000)
                    },
                },
            "genSumWeightName": "_nevents_",
            },
        ],
        "label":"MinBias_train",
        "color": cms10[0],
    },
    "MinBias_test" : {
        "groups" : [
            {
            "name": "MinBias_test",
            "samples": {
                "NuGunAllEta_PU200_test":
                    {
                    "xsec": 31038.96, #(2760.0 * 11246 / 1000)
                    },
                },
            "genSumWeightName": "_nevents_",
            },
        ],
        "label":"MinBias_test",
        "color": cms10[1],
    },
    "DoubleElectron_PU200_train" : {
        "groups" : [
            {
            "name": "DoubleElectron_PU200_train",
            "samples": {
                "DoubleElectron_FlatPt-1To100_PU200_train":
                    {
                    "xsec": None,
                    },
                },
            "genWeightName": None,
            "weight": "1",
            },
        ],
        "signal": True,
        "label":"DoubleEle_PU200_train",
        "color": cms10[2],
    },
    "DoubleElectron_PU200_test" : {
        "groups" : [
            {
            "name": "DoubleElectron_PU200_test",
            "samples": {
                "DoubleElectron_FlatPt-1To100_PU200_test":
                    {
                    "xsec": None,
                    },
                },
            "genWeightName": None,
            "weight": "1",
            },
        ],
        "signal": True,
        "label":"DoubleEle_PU200_test",
        "color": cms10[3],
    },
}

