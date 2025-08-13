#Signal train (1M produced indipendently)
run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/DoubleElePU200.py --flow flows/signal_flow.py   --plots plots/base_plots.py  -o tempSigTrain --eras "151Xv0pre4_TkElePtRegr_dev" --snapshot --columnSel "TkEle_(.*),nTkEle" --cache

#Signal test
run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/DoubleElePU200.py --flow flows/signal_flow.py   --plots plots/base_plots.py  -o tempSigTest --eras "151X_ptRegr_v0_A2" --snapshot --columnSel "TkEle_(.*),nTkEle" --cache

#Background train
run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/MinBias.py --flow flows/bkg_flow.py   --plots plots/bkg_plots.py  -o tempMinBias --eras "151X_ptRegr_v0_A2" --snapshot --columnSel "TkEle_(.*),nTkEle" --cache