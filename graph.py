import pandas as pd
import sys


class Graph:

    def __init__(self, num_vet=0, num_edg=0,mat_adj=None):
        self.num_vet = num_vet
        self.num_edg = num_edg

        if mat_adj is None:
            self.mat_adj = [[0 for _ in range(num_vet)] for _ in range(num_vet)]
        else:
            self.mat_adj = mat_adj

    def addEdge(self, source, destiny, capacity=float("inf"), flow=None) -> None:
        """

        :param flow: Flow value
        :param source: Source vertex
        :param destiny: Destiny vertex
        :param capacity: Capacity of the edge
        """
        if source < self.num_vet and destiny < self.num_vet:
            self.mat_adj[source][destiny] = (capacity, flow)
            self.num_edg += 1
        else:
            sys.exit("Invalid Edge")

    def removeEdge(self, source, destiny) -> None:
        """

        :param source: Source vertex
        :param destiny: Destiny vertex
        """
        if source < self.num_vet and destiny < self.num_vet:
            if self.mat_adj[source][destiny] != 0:
                self.num_edg -= 1
                self.mat_adj[source][destiny] = 0
        else:
            sys.exit("Invalid Edge")

    def getEdgesList(self) -> None:
        """

        :return: List of edges in format: source_vertex, destiny_vertex, (capacity, flow)
        """
        for i in enumerate(self.mat_adj):
            for j in enumerate(self.mat_adj[i]):
                edgesList = i, j, self.mat_adj[i][j]

        return edgesList

    @staticmethod
    def readFile(filename: str) -> None:
        """

        :param filename: Name of the csv file in /dataset
        :return: New status based on file data
        """
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")
            print(df.shape)  ## (rows, columns)
            print(df.to_string())
        except IOError:
            sys.exit("The file doesnt exists in /dataset")
