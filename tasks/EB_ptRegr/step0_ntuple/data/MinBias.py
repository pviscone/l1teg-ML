from cmgrdf_cli.data import cms10


all_processes={
    "MinBias" : {
        "groups" : [
            {
            "name": "MinBias",
            "samples": {
                "NuGunAllEta_PU200_train":
                    {
                    "xsec": None,
                    },
                },
            "genWeightName": None,
            "weight": "1",
            },
        ],
        "signal": False,
        "label":"MinBias PU200",
        "color": cms10[0],
    }
}

