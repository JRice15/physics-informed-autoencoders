import argparse
import re
import os
import matplotlib.pyplot as plt

from src.output_results import *

parser = argparse.ArgumentParser()

parser.add_argument("--name",required=True)
parser.add_argument("--file",required=True,help="file with names of test_results stats.txt files to read")

args = parser.parse_args()

# echo args
for k,v in args.__dict__.items():
    print("    " + k + ":", v, type(v))

os.makedirs("meta_results/" + args.name, exist_ok=True)

with open(args.file, "r") as f:
    filenames = f.readlines()

filenames = [i.strip() for i in filenames if i.strip() != ""]
filenames = [i.split(maxsplit=1) for i in filenames]

def allin(snippets, line):
    return all([i in line for i in snippets])

class TestData:

    def __init__(self, name, filename):
        self.filename = filename
        self.name = name

        with open(filename, "r") as f:
            lines = f.readlines()
        datalines = []
        in_files = True
        for i in lines:
            if not in_files:
                datalines.append(i)
            elif allin(["Min", "Avg", "Max", "Med", "Std"], i):
                in_files = False
        datalines = [i.strip().lower() for i in datalines]

        data = {}
        step_arr = []
        keys = ["min", "avg", "max", "med", "std"]
        for line in datalines:
            if line.startswith("step "):
                _, num = line.split()
                step_arr.append(int(num))
            else:
                line = line.split()
                try:
                    data[line[0]]
                except KeyError:
                    data[line[0]] = {k:[] for k in keys}
                for i in range(len(line[1:])):
                    data[line[0]][keys[i]].append( float(line[i+1]) )
                
        # data is dict of form: data[metric_name][min/avg/max/...] = list(values)
        self.data = data
        self.step_arr = step_arr

all_tests = []
for fname in filenames:

    all_tests.append(TestData(*fname))


def get_mark(data):
    data = [i[-1] for i in data]
    return data.index(min(data))

def met_name(met):
    if met == "mse":
        return "MSE"
    if met == "mae":
        return "MAE"
    if met == "relpred":
        return "Relative Error"
    return met.title()


metrics = all_tests[0].data.keys()
step_arr = all_tests[0].step_arr
dnames = [i.name for i in all_tests]

aggreg = "avg"

for met in metrics:
    for aggreg in ["min", "avg", "max", "med"]:
        plotdata = np.array([i.data[met][aggreg] for i in all_tests])

        mark = get_mark(plotdata)
        make_plot(xrange=step_arr, data=tuple(plotdata), dnames=dnames, title="Prediction " + aggreg.title() + " " + met_name(met), 
            mark=mark, axlabels=("steps", met_name(met)), legendloc="upper left",
            marker_step=step_arr[-1] - step_arr[-2], fillbetweens=None,
            fillbetween_desc="", ylim=None, ymin=0)
        name = aggreg + "_" + met
        print(name)
        plt.savefig("meta_results/" + args.name + "/" + name + ".png")
        plt.clf()

        if aggreg in ["avg", "med"]:
            stds = np.array([i.data[met]["std"] for i in all_tests])
            low = plotdata - stds
            high = plotdata + stds
            fillbetweens = [(low[i], high[i]) for i in range(len(low))]
            make_plot(xrange=step_arr, data=tuple(plotdata), dnames=dnames, title="Prediction " + aggreg.title() + " " + met_name(met), 
                mark=mark, axlabels=("steps", met_name(met)), legendloc="upper left",
                marker_step=step_arr[-1] - step_arr[-2], fillbetweens=fillbetweens,
                fillbetween_desc="with 1 std. dev.", ylim=None, ymin=0)
            name = aggreg + "_" + met + ".w_stds"
            print(name)
            plt.savefig("meta_results/" + args.name + "/" + name + ".png")
            plt.clf()

