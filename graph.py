import pandas as pd
import sys


class Graph:

    def __init__(self, num_vet=0, num_edg=0, mat_adj=None):
        self.num_vet = num_vet
        self.num_edg = num_edg

        if mat_adj is None:
            self.mat_adj = [[0 for _ in range(num_vet)] for _ in range(num_vet)]
        else:
            self.mat_adj = mat_adj

    @staticmethod
    def readFile(filename):
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")
            print(df.shape) ## (rows, columns)
            print(df.to_string())
        except IOError:
            sys.exit("The file doesnt exists in /dataset")
