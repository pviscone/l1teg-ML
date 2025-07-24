run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/DoubleElePU0.py  --flow flows/base_flow.py   --plots plots/base_plots.py  -o tempPU0 --eras "140Xv0C1" --cache
run_analysis.py --cfg cfg/l1teg_cfg.py  --mc data/DoubleElePU200.py  --flow flows/base_flow.py   --plots plots/base_plots.py  -o tempPU200 --eras "151Xv0pre4_TkElePtRegr_dev_withScaled" --snapshot --columnSel "TkEle_(.*),nTkEle" --cache
