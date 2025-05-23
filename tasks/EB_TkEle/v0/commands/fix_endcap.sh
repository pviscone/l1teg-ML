run_analysis.py --cfg cfg/fix_endcap.py  --mc data/samples.py --flow flows/matchOnly.py:region=\"EE\"   --plots plots/minimal.py  -o temp --eras v131Xv3O,v131Xv9A,140Xv0B18 --noStack --snapshot --columnSel "GenEl_pt,TkEleL2_idScore,TkEleL2_pt,TkEleL2_GenEl_pt,TkEleL2_genIdx,weight" --eraSel "140Xv0B18"

python scripts/eff_rate.py -c scripts/effrate_cfg/fix_endcap.yaml -i temp