from CMGRDF.cms.eras import lumis as lumi

P = "/eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_ntuples"

base_tuple = (P, "{name}/FP/{era}/*.root", "")

era_paths_Data = {}
era_paths_MC = {}

PFs = []
PMCs = []

def add_tag(tags):
    for tag in tags:
        era_paths_Data[tag] = base_tuple
        era_paths_MC[tag] = base_tuple
        lumi[tag] = 0.001

add_tag(["151Xv0pre4_TkElePtRegr_dev", "140Xv0C1", "151Xv0pre4_TkElePtRegr_dev_withScaled"])

