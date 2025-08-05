#Big training dataset
run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/DoubleElePU200.py --flow flows/base_flow.py   --plots plots/base_plots.py  -o tempTrain --eras "151Xv0pre4_TkElePtRegr_dev_withScaled" --snapshot --columnSel "TkEle_(.*),nTkEle" --cache

#Test Dataset with corrected pt
run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/DoubleElePU200.py --flow flows/base_flow.py   --plots plots/base_plots.py  -o tempA2 --eras "151X_ptRegr_v0_A2" --snapshot --columnSel "TkEle_(.*),nTkEle" --cache