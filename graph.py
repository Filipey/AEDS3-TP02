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

                self.num_edg += 1
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

    def cleanSubjects(self, subjects) -> list:
        """

        :param subjects: list of subjects with NaN values to be cleaned
        :return: new list with subjects
        """

        new_subjects = []
        for item in subjects:
            new_subjects.append([subject for subject in item if str(subject) != 'nan'])

        new_subjects.pop(-1)

        return new_subjects

    def readTeachers(self, filename: str) -> tuple:
        """

        :param filename: Name of the teachers csv file in /dataset
        :return: New graph status based on file data
        """
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")

            teachers = df.iloc[:, 0].dropna().values.tolist()  # Get values of column Professor, removing NaN values

            subjects_offered = df.iloc[:, 1].values.tolist()  # Get the subjects offered of each teacher
            subjects_offered.pop(-1)  # Removed unused total of subjects offered

            subjects = df.iloc[:, [2, 3, 4, 5, 6]].values.tolist()  # Get values of columns Preferencia's,

            subjects = self.cleanSubjects(subjects)  # Clean subjects list

            return teachers, subjects_offered, subjects

        except IOError:
            sys.exit("The file doesnt exists in /dataset")

    def readSubjects(self, filename: str) -> tuple:
        """

        :param filename: Name of the subjects csv file in /dataset
        :return: New graph status based on file data
        """
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")

            data = df.iloc[:, ].values.tolist()  # Get all data into a list

            num_of_classes = data.pop(-1)[2]  # Remove last line that contains the total of classes

            total_of_subjects = len(df.iloc[:, 0].dropna().values.tolist())  # Get the total of subjects offered

            return data, num_of_classes, total_of_subjects

        except IOError:
            sys.exit("The file doesnt exists in /dataset")

    def setOriginEdges(self, teachers: list, subjects_offered: list) -> None:
        for i in range(0, len(teachers)):
            origin = self.mat_adj[0][1]
            destiny_teacher = self.mat_adj[0][i]
            teacher_capacity = subjects_offered[i]
            self.addEdge(origin, destiny_teacher, teacher_capacity)

    def setDestinyEdges(self, initial_vertex: int, subjects: list, subjects_info: list) -> None:
        subjects_capacities = [c[2] for c in subjects_info]

        for i in range(initial_vertex, self.num_vet - 1):
            destiny = self.mat_adj[self.num_vet - 1][0]
            origin_subject = self.mat_adj[self.num_vet - 1][i]
            for c in subjects_capacities:
                subject_capacity = c
                subjects_capacities.remove(c)
                break
            self.addEdge(origin_subject, destiny, subject_capacity)

    def setInitialData(self, teachers_data: tuple, subjects_data: tuple):
        (teachers, subjects_offered, subjects) = teachers_data
        (subjects_info, num_of_classes, total_of_subjects) = subjects_data

        # updating num_vet and num_edg based on files data
        self.num_vet = 2 + len(teachers) + total_of_subjects
        self.num_edg = len(teachers) + total_of_subjects + num_of_classes

        # updating data structures with new data
        self.mat_adj = [[0 for _ in range(self.num_vet)] for _ in range(self.num_vet)]
        self.list_adj = [[] for _ in range(self.num_vet)]

        # flow value based on the preferences table
        flow = [0, 3, 5, 8, 10]

        # adding edge from origin 's' to each teacher
        # with capacity equals to their subjects_offered
        self.setOriginEdges(teachers, subjects_offered)

        # adding edge from each subject to destiny 't'
        # with capacity equals to number of classes of the subject
        self.setDestinyEdges(len(teachers) + 1, subjects, subjects_info)

    def bellmanFord(self, s, v):
        dist = [float("inf") for _ in range(len(self.list_adj))]
        pred = [None for _ in range(len(self.list_adj))]
        edges = self.getEdgesList()

        dist[s] = 0

        for i in range(0, len(self.list_adj) - 1):
            trade = False
            for [source, destiny, (flow, capacity)] in edges:  # edge = [source, destiny, (flow, capacity)]
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

    def run(self, teachers_file: str, subjects_file: str) -> None:
        self.setInitialData(self.readTeachers(teachers_file), self.readSubjects(subjects_file))
