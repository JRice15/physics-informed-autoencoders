import argparse
import os
import string

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name",required=True)
parser.add_argument("-f","--file",required=True)
parser.add_argument("-t","--time",required=True,type=str)
parser.add_argument("--seperate",action="store_true",default=False,help="split file lines into a task per line")

args = parser.parse_args()

slurmdir = os.path.dirname(os.path.realpath(__file__))

# create dirs
os.makedirs(os.path.join(slurmdir, "logs"), exist_ok=True)
os.makedirs(os.path.join(slurmdir, "tasks"), exist_ok=True)

# read template
template = os.path.join(slurmdir, "template.sh")
with open(template, "r") as f:
    template_lines = f.readlines()

# read input test file
with open(args.file, "r") as f:
    test_lines = f.readlines()


def maybe_format(line, write_name, time_str):
    """
    format if it has format fields. uses a nifty hack from SO:
    https://stackoverflow.com/questions/46161710/how-to-check-if-string-has-format-arguments-in-python
    """
    field_names = [tup[1] for tup in string.Formatter().parse(line) if tup[1] is not None]
    if "logname" in field_names:
        line = line.format(logname=write_name)
    if "time" in field_names:
        line = line.format(time=time_str)
    return line

def do_writing(template_lines, test_lines, index=None):
    """
    write a task file
    """
    write_name = args.name + "_task"
    if index is not None:
        write_name += "_" + str(index)

    new_template_lines = [maybe_format(i, write_name, args.time) for i in template_lines]

    filename = write_name + ".slurm"

    task_file = os.path.join(slurmdir, "tasks", filename)
    if os.path.exists(task_file):
        prompt = "A task with name '{}' exists. Overwrite? [y/n]: ".format(filename)
        if input(prompt).strip().lower() not in ("y", "yes"):
            print("skipping...")
            return
        print("overwriting...")

    full_lines = new_template_lines + test_lines
    with open(task_file, "w") as f:
        f.writelines(full_lines)


if args.seperate:
    test_lines = [i for i in test_lines if i.strip() != ""]
    for i in range(len(test_lines)):
        # write non-empty lines as their own task
        do_writing(template_lines, [test_lines[i]], i)
else:
    # write all lines to a task
    do_writing(template_lines, test_lines)
