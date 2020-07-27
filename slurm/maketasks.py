import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("file")
parser.add_argument("--seperate",action="store_true",default=False,help="split file lines into a task per line")

args = parser.parse_args()

slurmdir = os.path.dirname(os.path.realpath(__file__))

# read template
template = os.path.join(slurmdir, "template.sh")
with open(template, "r") as f:
    template_lines = f.readlines()

# read input test file
with open(args.file, "r") as f:
    test_lines = f.readlines()


def do_writing(lines, index=None):
    """
    write a task file
    """
    filename = args.name + "_task"
    if index is not None:
        filename += "_" + str(index)
    filename += ".slurm"
    task_file = os.path.join(slurmdir, "tasks", filename)
    if os.path.exists(task_file):
        prompt = "a task with name '{}' exists. overwrite? [y/n]: ".format(filename)
        if input(prompt).strip().lower() not in ("y", "yes"):
            return
    full_lines = template_lines + test_lines
    with open(task_file, "w") as f:
        f.writelines(full_lines)


if args.seperate:
    for i in range(len(test_lines)):
        # write non-empty lines as their own task
        if test_lines[i].strip() != "":
            do_writing([test_lines[i]], i)
else:
    # write all lines to a task
    do_writing(test_lines)
