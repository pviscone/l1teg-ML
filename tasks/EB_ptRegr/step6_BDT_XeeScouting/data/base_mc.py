from cmgrdf_cli.data import cms10

all_processes={
    "MinBias" : {
        "groups" : [
            {
            "name": "MinBias",
            "samples": {
                "NuGunAllEta_PU200":
                    {
                    "xsec":31038.96*1000*60*24*60*60, #(2760.0 * 11246 / 1000) RATE (ev per sec / 1000) *1000 * 60days (equivalent to 400fb-1)
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
    "XeeM2" : {
        "groups" : [
            {
            "name": "XeeM2",
            "samples": {
                "ZdToEE_HAHM_012j_14TeV_PU200_M2":
                    {
                    "xsec":10,
                    },
                },
            "genSumWeightName": "_nevents_",
            "cut":"1",
            },
        ],
        "label":"Xee M2",
        "signal": True,
        "color": cms10[1],
    },
    "XeeM5" : {
        "groups" : [
            {
            "name": "XeeM5",
            "samples": {
                "ZdToEE_HAHM_012j_14TeV_PU200_M5":
                    {
                    "xsec":10,
                    },
                },
            "genSumWeightName": "_nevents_",
            "cut":"1",
            },
        ],
        "label":"Xee M5",
        "signal": True,
        "color": cms10[2],
    },
    "XeeM10" : {
        "groups" : [
            {
            "name": "XeeM10",
            "samples": {
                "ZdToEE_HAHM_012j_14TeV_PU200_M10":
                    {
                    "xsec":10,
                    },
                },
            "genSumWeightName": "_nevents_",
            "cut":"1",
            },
        ],
        "label":"Xee M10",
        "signal": True,
        "color": cms10[3],
    },
    "XeeM15" : {
        "groups" : [
            {
            "name": "XeeM15",
            "samples": {
                "ZdToEE_HAHM_012j_14TeV_PU200_M15":
                    {
                    "xsec":10,
                    },
                },
            "genSumWeightName": "_nevents_",
            "cut":"1",
            },
        ],
        "label":"Xee M15",
        "signal": True,
        "color": cms10[4],
    },
    "XeeM20" : {
        "groups" : [
            {
            "name": "XeeM20",
            "samples": {
                "ZdToEE_HAHM_012j_14TeV_PU200_M20":
                    {
                    "xsec":10,
                    },
                },
            "genSumWeightName": "_nevents_",
            "cut":"1",
            },
        ],
        "label":"Xee M20",
        "signal": True,
        "color": cms10[5],
    },
    "XeeM30" : {
        "groups" : [
            {
            "name": "XeeM30",
            "samples": {
                "ZdToEE_HAHM_012j_14TeV_PU200_M30":
                    {
                    "xsec":10,
                    },
                },
            "genSumWeightName": "_nevents_",
            "cut":"1",
            },
        ],
        "label":"Xee M30",
        "signal": True,
        "color": cms10[6],
    },    
}
