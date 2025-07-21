from cmgrdf_cli.data import cms10

from cmgrdf_cli.data import cms10

def all_processes():
    return {
    "DoublePhoton_PU0" : {
        "groups" : [
            {
            "name": "DoublePhoton_PU0",
            "samples": {
                "DoublePhoton_FlatPt-1To100_PU0":
                    {
                    "xsec": None,
                    },
                },
            "genWeightName": None,
            "weight": "1",
            },
        ],
        "signal": True,
        "label":"DoublePhoton_PU0",
        "color": cms10[0],
    },
    "DoubleElectron_PU0" : {
        "groups" : [
            {
            "name": "DoubleElectron_PU0",
            "samples": {
                "DoubleElectron_FlatPt-1To100_PU0":
                    {
                    "xsec": None,
                    },
                },
            "genWeightName": None,
            "weight": "1",
            },
        ],
        "signal": True,
        "label":"DoubleEle_PU0",
        "color": cms10[1],
    },
}

