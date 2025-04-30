# TO run only on samples that will be used for training (140Xv0B9)

#Save only the cluster pt histogram
if [[ "$*" != *"--no_pt_hist"* ]]; then
    run_analysis.py --cfg cfg/l1teg_cfg.py --mc data/TrainTest_samples.py  --flow flows/Barrel/TkEle/FlowBase.py  --plots plots/Barrel/TkEle/PlotTkEleFeatures.py:pt_only=True  -o $2 --eras $1 --noYields --noStack
fi

#Read the cluster pt histogram to reweight the TkEle features and save a snapshot
run_analysis.py --cfg cfg/l1teg_cfg.py --mc data/TrainTest_samples.py  --flow flows/Barrel/TkEle/FlowPtReweight.py:pt_hist=\"$2/era$1/matching_1_full/TkEle_CryClu_pt.root,DoubleElectron_PU200_train,MinBias_train\"  --plots plots/Barrel/TkEle/PlotTkEleFeatures.py  -o $2 --eras $1 --snapshot --columnSel "TkEle_.*" --noStack