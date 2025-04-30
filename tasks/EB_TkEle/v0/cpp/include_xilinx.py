import os
import glob
from cmgrdf_cli.cpp import add_include_path, include

def declare(*paths):
    this_dir = os.path.dirname(__file__)
    add_include_path(os.path.join(this_dir, "conifer"))
    add_include_path(os.path.join(this_dir, "conifer/Vitis_HLS/simulation_headers/include"))
    for path in paths:
        path = os.path.join(this_dir, path)
        add_include_path(path)
        for filepath in glob.glob(os.path.join(path, "*")):
            include(filepath)