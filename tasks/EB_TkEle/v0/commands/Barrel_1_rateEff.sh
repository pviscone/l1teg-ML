#Save only the cluster pt histogram
if [[ "$*" != *"--no_baseline"* ]]; then
    run_analysis.py --cfg cfg/l1teg_cfg.py --mc data/samples.py   --flow flows/Barrel/Baseline/FlowBaseline.py   --plots plots/Barrel/Baseline/PlotEffRate.py  -o $1 --eras 140Xv0B6,140Xv0B9 --noStack
fi

run_analysis.py --cfg cfg/l1teg_cfg.py --mc data/TrainTest_samples.py  --flow flows/Barrel/TkEle/FlowBDT.py:bdt_path=\"/eos/user/p/pviscone/www/L1T/l1teg/EB_TkEle/v0/zmodels/EB_TkEleID/xgb_2class/v1\"  --plots plots/Barrel/TkEle/PlotTkEleFeatures.py:score_only=True -o $1 --eras 140Xv0B9 --noStack --snapshot --columnSel "TkEle_score,TkEle_(GenEle|CryClu)_pt,TkEle_GenEle_eta,TkEle_(Tk|GenEle)_idx,weight"