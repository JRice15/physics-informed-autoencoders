import argparse
import os
import string

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


def do_writing(template_lines, test_lines, index=None):
    """
    write a task file
    """
    write_name = args.name + "_task"
    if index is not None:
        write_name += "_" + str(index)

    is_fmt = lambda line: len(tuple(string.Formatter().parse(line))) > 1
    new_template_lines = [i.format(logname=write_name) if is_fmt(i) else i for i in template_lines]

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
