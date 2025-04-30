from cmgrdf_cli.data import cms10

def all_processes():
    return {
    "MinBias" : {
        "groups" : [
            {
            "name": "MinBias",
            "samples": {
                "NuGunAllEta_PU200":
                    {
                    "xsec": 31038.96, #(2760.0 * 11246 / 1000)
                    },
                },
            "genSumWeightName": "_nevents_",
            "weight": "1.",
            },
        ],
        "label":"MinBias",
        "color": cms10[0],
    },
    "DoubleElectron_PU200" : {
        "groups" : [
            {
            "name": "DoubleElectron_PU200",
            "samples": {
                "DoubleElectron_FlatPt-1To100_PU200":
                    {
                    "xsec": None,
                    },
                },
            "genWeightName": None,
            "weight": "1",
            },
        ],
        "signal": True,
        "label":"DoubleEle_PU200",
        "color": cms10[1],
    },
}

