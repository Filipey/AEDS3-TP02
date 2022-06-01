import pandas as pd
import sys


class Graph:

    def __init__(self, num_vet=0, num_edg=0, mat_adj=None, list_adj: list = None):
        self.num_vet = num_vet
        self.num_edg = num_edg

        if mat_adj is None:
            self.mat_adj = [[0 for _ in range(num_vet)] for _ in range(num_vet)]
        else:
            self.mat_adj = mat_adj

        if list_adj is None:
            self.list_adj = [[] for _ in range(num_vet)]
        else:
            self.list_adj = list_adj

    def addEdge(self, source, destiny, capacity=float("inf"), flow=None) -> None:
        """

        :param flow: Flow value
        :param source: Source vertex
        :param destiny: Destiny vertex
        :param capacity: Capacity of the edge
        """
        if source < self.num_vet and destiny < self.num_vet:
            self.mat_adj[source][destiny] = (flow, capacity)
            self.list_adj[source].append((destiny, (flow, capacity)))
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

                for (v, w) in self.list_adj[source]:
                    if v == destiny:
                        self.list_adj[source].remove((v, w))
                        break
        else:
            sys.exit("Invalid Edge")

    def getEdgesList(self) -> list:
        """

        :return: List of edges in format: source_vertex, destiny_vertex, (flow, capacity)
        """
        edges_list = []

        for i in enumerate(self.mat_adj):
            for j in enumerate(self.mat_adj[i]):
                edges_list.append((i, j, self.mat_adj[i][j]))

        return edges_list

    @staticmethod
    def readFile(filename: str) -> None:
        """

        :param filename: Name of the csv file in /dataset
        :return: New status based on file data
        """
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")
            print(df.shape)  # (rows, columns)
            print(df.to_string())
        except IOError:
            sys.exit("The file doesnt exists in /dataset")

    def bellmanFord(self, s, v):
        dist = [float("inf") for _ in range(len(self.list_adj))]
        pred = [None for _ in range(len(self.list_adj))]
        edges = self.getEdgesList()

        dist[s] = 0

        for i in range(0, len(self.list_adj) - 1):
            trade = False
            for edge in edges:  # edge = [source, destiny, (flow, capacity)]
                source = edge[0]
                destiny = edge[1]
                flow = edge[2][0]

                if dist[destiny] > dist[source] + flow:
                    dist[destiny] = dist[source] + flow
                    pred[destiny] = source

                    trade = True

            if trade is False:
                break

        shortest_path = [v]
        i = pred[v]
        while i in pred:
            if i is None:
                break
            shortest_path.append(i)
            i = pred[i]

        return shortest_path
