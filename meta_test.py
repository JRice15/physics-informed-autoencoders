import argparse
import re
import os

import scipy.stats as scipy_stats

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

        self.num_seeds = 0

        with open(filename, "r") as f:
            lines = f.readlines()
        datalines = []
        in_files = True
        for i in lines:
            if not in_files:
                datalines.append(i)
            elif allin(["Min", "Avg", "Max", "Med", "Std"], i):
                in_files = False
            elif i.strip() != "":
                self.num_seeds += 1
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


def conf_interval_90(means, stds, n):
    """
    90 percent confidence interval
    X_bar +/- Z (std / sqrt(n))
    """
    Z = 1.645
    n = n.reshape(-1, 1)
    err = Z * stds / np.sqrt(n)
    return means - err, means + err, err


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


# PLOTTING
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
        plt.savefig("meta_results/" + args.name + "/" + args.name + "." + name + ".png")
        plt.clf()

        if aggreg in ["avg", "med"]:
            stds = np.array([i.data[met]["std"] for i in all_tests])
            n = np.array([i.num_seeds for i in all_tests])
            low, high, error = conf_interval_90(plotdata, stds, n)
            fillbetweens = [(low[i], high[i]) for i in range(len(low))]
            make_plot(xrange=step_arr, data=tuple(plotdata), dnames=dnames, title="Prediction " + aggreg.title() + " " + met_name(met), 
                mark=mark, axlabels=("steps", met_name(met)), legendloc="upper left",
                marker_step=step_arr[-1] - step_arr[-2], fillbetweens=fillbetweens,
                fillbetween_desc="with 90% confidence interval", ylim=None, ymin=0)
            name = aggreg + "_" + met + ".w_90per_conf"
            print(name)
            plt.savefig("meta_results/" + args.name + "/" + args.name + "." + name + ".png")
            plt.clf()

        if aggreg == "avg":
            # confidence interval bar graph
            plotdata = [(i.data[met][aggreg][-1], i.name) for i in all_tests]

            plotdata = [plotdata[i] + (error[i][-1],) for i in range(len(plotdata))]
            plotdata.sort(key=lambda x: x[0])

            data = [i[0] for i in plotdata]
            names = [i[1] for i in plotdata]
            err = [i[2] for i in plotdata]

            for i in range(len(data)):
                color = "C" + str(i)
                bar = plt.bar(names[i], data[i], alpha=0.2, ec="black", 
                    ecolor=color, 
                    error_kw={"capthick": 1.5},
                    yerr=err[i], capsize=3000)
                bar[0].set_color(color)

            plt.title("Step 180 " + aggreg.title() + " " + met_name(met))
            plt.xticks(rotation=10)
            plt.tight_layout()
            # plt.show()
            plt.savefig("meta_results/" + args.name + "/" + args.name + "." + met + "_confidence.png")
            plt.clf()


            # write confidences
            with open("meta_results/" + args.name + "/" + args.name + "." + met + "_confidence.txt", "w") as f:
                # conf interval on difference
                f.write("\t\t")
                for t in all_tests:
                    f.write(t.name + "\t")
                for test in all_tests:
                    f.write("\n" + test.name + "\t")
                    for othertest in all_tests:
                        if othertest != test:
                            mean1 = test.data[met][aggreg][-1]
                            mean2 = othertest.data[met][aggreg][-1]
                            std1 = test.data[met]["std"][-1]
                            std2 = othertest.data[met]["std"][-1]
                            n1 = test.num_seeds
                            n2 = othertest.num_seeds
                            
                            mean_diff = abs(mean1 - mean2)
                            error_thing = np.sqrt( (std1**2)/n1 + (std2**2)/n2 )

                            Z = mean_diff / error_thing

                            p = scipy_stats.norm.cdf(Z)
                            p = (2 * p) - 1 # convert to two-sided p value
                            pct = p * 100

                            # print(test.name, "\tvs\t", othertest.name, "\t", pct)
                            f.write("{:5.3f}\t".format(pct))
                        else:
                            f.write("x    \t")


