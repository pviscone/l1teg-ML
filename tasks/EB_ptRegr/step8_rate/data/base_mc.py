from cmgrdf_cli.data import cms10


all_processes={
    "MinBias" : {
        "groups" : [
            {
            "name": "MinBias",
            "samples": {
                "NuGunAllEta_PU200":
                    {
                    "xsec": 31038.96, #(2760.0 * 11246 / 1000) RATE (ev per sec / 1000) *1000 * 60days (equivalent to 400fb-1)
                    },
                },
            "genSumWeightName": "_nevents_",
            "weight": "1.",
            "cut":"1",
            },
        ],
        "label":"MinBias",
        "signal": False,
        "color": cms10[0],
    },
}
