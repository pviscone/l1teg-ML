import os
import sys
import copy

sys.path.append(os.environ["PWD"])

import typer
import yaml
from typing import Annotated
import re

from cmgrdf_cli.plots.plotters import TEfficiency, TRate
from eff_rate.efficiency import _plot_efficiency_normal, _plot_efficiency_varbins
from eff_rate.rates import _plot_rate_normal, _plot_rate_varbins

app = typer.Typer(
    pretty_exceptions_show_locals=False, rich_markup_mode="rich", add_completion=False
)


def _plot_eff_rate(cfg, name, output, plot_dict, obj_pattern, lines):
    objs = plot_dict["items"]
    base_path = cfg["base_path"]
    objs_map = cfg["objs_map"]
    rate_kwargs = plot_dict.get("rate_kwargs", {})
    tRate = TRate(cmstext="Phase-2 Simulation Preliminary",
        lumitext="PU 200 (14 TeV)",
        grid=False, **rate_kwargs)
    for _ivar, variable in enumerate(cfg["vars"]):
        variable_name = variable
        variable_dict = copy.deepcopy(cfg["vars"][variable])
        snapBranch = variable_dict.pop("snapBranch", None)
        tEff = TEfficiency(
            ylabel="Efficiency",
            **variable_dict,
            cmstext="Phase-2 Simulation Preliminary",
            lumitext="PU 200 (14 TeV)",
            grid=False,
        )
        tEff.add_line(y=1, linestyle="--", alpha=0.15, linewidth=1, color="black")
        tEff.add_line(y=0.9, linestyle="--", alpha=0.15, linewidth=1, color="black")
        for obj in objs:
            for pattern in obj_pattern.split(","):
                if re.match(pattern, obj):
                    global_tag = objs_map[obj]["global_tag"]
                    if "Eff_num" in objs_map[obj] and "Eff_den" in objs_map[obj]:
                        eff_num = objs_map[obj]["Eff_num"].format(
                            global_tag=global_tag, variable=variable_name
                        )
                        eff_den = objs_map[obj]["Eff_den"].format(
                            global_tag=global_tag, variable=variable_name
                        )

                        if ":" in eff_num:
                            path_num, branch_num = eff_num.split(":")
                            path_num = os.path.join(base_path, path_num)

                            path_den, branch_den = eff_den.split(":")
                            path_den = os.path.join(base_path, path_den)

                            tEff = _plot_efficiency_normal(
                                tEff, obj, path_num, branch_num, path_den, branch_den
                            )
                        else:
                            path_num = os.path.join(base_path, eff_num)
                            path_den, branch_den = eff_den.split(":")
                            path_den = os.path.join(base_path, path_den)

                            score = objs_map[obj]["score"]
                            variable = snapBranch
                            genidx = objs_map[obj]["genIdx"]
                            binVar = objs_map[obj]["binVar"]

                            bins = objs_map[obj]["Bins"]
                            thrs = objs_map[obj]["Thrs"]
                            if lines:
                                for b in bins:
                                    tEff.add_line(
                                        x=b,
                                        linestyle="--",
                                        alpha=0.2,
                                        linewidth=1,
                                        color="red",
                                    )

                            tEff = _plot_efficiency_varbins(
                                tEff,
                                obj,
                                path_num,
                                path_den,
                                branch_den,
                                score,
                                variable,
                                genidx,
                                binVar,
                                bins,
                                thrs,
                            )
                    if "Rate" in objs_map[obj] and _ivar==0:
                        rate = objs_map[obj]["Rate"].format(global_tag=global_tag)
                        if ":" in rate:
                            path_rate, branch_rate = rate.split(":")
                            path_rate = os.path.join(base_path, path_rate)
                            _plot_rate_normal(tRate, obj, path_rate, branch_rate)
                        else:
                            if lines:
                                for b in bins:
                                    tRate.add_line(
                                        x=b,
                                        linestyle="--",
                                        alpha=0.2,
                                        linewidth=1,
                                        color="red",
                                    )
                            path = os.path.join(base_path, rate)
                            score = objs_map[obj]["score"]
                            rateVar = objs_map[obj]["rateVar"]
                            binVar = objs_map[obj]["binVar"]
                            bins = objs_map[obj]["Bins"]
                            thrs = objs_map[obj]["Thrs"]
                            _plot_rate_varbins(tRate, obj, path, score, rateVar, binVar, bins, thrs)

        os.makedirs(os.path.join(output, f"{variable_name}"), exist_ok=True)
        tEff.save(os.path.join(output, f"{variable_name}/{name}_eff.pdf"))
        tEff.save(os.path.join(output, f"{variable_name}/{name}_eff.png"))
    tRate.save(os.path.join(output, f"{name}_rate.pdf"))
    tRate.save(os.path.join(output, f"{name}_rate.png"))

import multiprocessing as mp
@app.command()
def plot_eff_rate(
    cfg: Annotated[str, typer.Option("-c", "--cfg", help="Path to the yaml conf file")],
    output: Annotated[str, typer.Option("-o", "--output", help="Output directory")],
    obj_pattern: str = typer.Option(
        ".*", "--objs", help="Objects to plot (regex pattern, comma separater)"
    ),
    ncpu: int = typer.Option(
        mp.cpu_count(), "-j", "--ncpus", help="Number of cpus to use for plotting"
    ),
    plot_pattern: str = typer.Option(
        ".*", "--plots", help="Plots to plot (regex pattern, comma separater)"
    ),
    lines : bool = typer.Option(
        False, "--lines", help="Add lines to the plots"
    ),
):
    os.makedirs(output, exist_ok=True)
    os.system(f"cp -f {cfg} {output}")
    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
    plots = cfg["plots"]
    pool_data = []
    for plot in plots:
        if not re.match(plot_pattern, plot):
            continue
        pool_data.append((cfg, plot, output, plots[plot], obj_pattern, lines))
    if ncpu > 1:
        with mp.Pool(ncpu) as p:
            p.starmap(_plot_eff_rate, pool_data, chunksize=max(1, len(pool_data)//mp.cpu_count()))
    else:
        for d in pool_data:
            _plot_eff_rate(*d)


if __name__ == "__main__":
    app()
