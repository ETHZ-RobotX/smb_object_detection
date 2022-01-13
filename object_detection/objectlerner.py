import os 
import sys
import yaml
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
CFG_DIR     = os.path.join(SCRIPT_DIR,"cfg")
DATA_DIR    = os.path.join(CFG_DIR,"data")


def bb2dist(degree = 5):

    bb2dist_dict = {}
    data_dir = os.path.join(DATA_DIR, "bb2dist")

    for filename in os.listdir(data_dir):
        obj, _ = os.path.splitext(filename)
        data = np.loadtxt( os.path.join(data_dir,filename) )
        bb2dist_dict[obj] = np.polyfit(data[:,0], data[:,1], degree).tolist()

    with open(os.path.join(CFG_DIR,"bb2dist.yaml"), 'w') as file:
        yaml.dump(bb2dist_dict, file)

if __name__=='__main__':

    learner_type = sys.argv[0]

    if learner_type == "bb2dist":
        bb2dist()
    else:
        bb2dist()
