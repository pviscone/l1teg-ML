from CMGRDF.cms.eras import lumis as lumi

P = "/eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_perfTuple"

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

#140Xv0B6 contains the baseline objects (Barrel EllipticId, Endcap ???)
add_tag(["140Xv0B6", "140Xv0B9", "140Xv0C1"])

