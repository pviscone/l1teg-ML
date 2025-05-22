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

#140Xv0B6 contains the baseline objects (Barrel EllipticId, Endcap ???)
#v131Xv3O is AR2024
#v131Xv9A is DPS-note (new hgcal id)
#140Xv0B12 is 142X-int (boh)
#140Xv0B18 is 142X-int-gct (new hgcal id + unpacker)
add_tag(["v131Xv3O", "v131Xv9A", "140Xv0B18", "140Xv0C1"])
