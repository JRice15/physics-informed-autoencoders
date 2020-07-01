

class Defaults:
    """
    namespace for defining arg defaults
    """
    lr = 0.001
    epochs = 6000
    batchsize = 34
    lambda_ = 5 # inverse regularizer weight
    kappa = 1 # stability regularizer weight
    gamma = 4 # stability regularizer steepness
    sizes = (40, 25, 15) # largest to smallest

def get_run_name(args):
    run_name = ""
    if args.tag is not None:
        run_name += args.tag + "."
    if args.no_stability:
        run_name += "nostability."
    else:
        run_name += "stability."
    run_name += "{}epochs_{}batchsize_{}lr.".format(args.epochs, args.batchsize, args.lr)
    run_name += "s{}_{}_{}.".format(args.s1, args.s2, args.s3)
    run_name += "l{}_k{}_g{}".format(args.lambd, args.kappa, args.gamma)
    return run_name
