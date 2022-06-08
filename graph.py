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

        self.subjects_index = {}
        self.teachers_index = {}

    def addEdge(self, source, destiny, capacity=float("inf"), flow=None) -> None:
        """
        Add an edge on graph in format (source, destiny, (flow, capacity))

        :param flow: Flow value
        :param source: Source vertex
        :param destiny: Destiny vertex
        :param capacity: Capacity of the edge
        """
        if source < self.num_vet and destiny < self.num_vet:
            self.mat_adj[source][destiny] = [flow, capacity]
            self.list_adj[source].append((destiny, [flow, capacity]))
            self.num_edg += 1
        else:
            sys.exit("Invalid Edge")

    def removeEdge(self, source, destiny) -> None:
        """
        Delete a edge from graph

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
        Get list of edges in format: source_vertex, destiny_vertex, (flow, capacity)

        :return: List of edges
        """
        edges_list = []

        for i in range(0, len(self.mat_adj)):
            for j in range(0, len(self.mat_adj[i])):
                if self.mat_adj[i][j] != 0:
                    edges_list.append((i, j, self.mat_adj[i][j]))

        return edges_list

    @staticmethod
    def cleanSubjects(subjects) -> list:
        """
        Removed unusual data from subjects list

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
        Read teachers file and return formatted data

        :param filename: Name of the teachers csv file in /dataset
        :return: teachers: list, subjects_offered: list, subjects: list
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

    @staticmethod
    def readSubjects(filename: str) -> tuple:
        """
        Read subjects file and return formatted data

        :param filename: Name of the subjects csv file in /dataset
        :return: subjects_info: list, num_of_classes: int, total_of_subjects: list
        """
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")

            data = df.iloc[:, ].values.tolist()  # Get all data into a list

            num_of_classes = data.pop(-1)[2]  # Remove last line that contains the total of classes

            total_of_subjects = len(df.iloc[:, 0].dropna().values.tolist())  # Get the total of subjects offered

            return data, num_of_classes, total_of_subjects

        except IOError:
            sys.exit("The file doesnt exists in /dataset")

    def setTeachersAndSubjectsIndexes(self, subjects_initial_vertex: int, subjects_info: list, teachers_data: tuple):
        """
        Set the key/value of each teacher and subject

        :param teachers_data: tuple with data of each teacher
        :param subjects_initial_vertex: initial vertex of subjects
        :param subjects_info: list of each subject info
        :return:
        """
        (teachers, num_of_subject_offered, subjects_offered) = teachers_data

        for j in range(subjects_initial_vertex, self.num_vet - 1):
            for subject in subjects_info:
                self.subjects_index[j] = subject
                subjects_info.remove(subject)
                break

        for i in range(0, len(teachers)):
            self.teachers_index[i+1] = (teachers[i], num_of_subject_offered[i], subjects_offered[i])

    def setOriginEdges(self, teachers: list, subjects_offered: list) -> None:
        """
        Set edges from origin vertex to teachers

        :param teachers: List of teachers
        :param subjects_offered: List of subjects that each teacher apply
        :return: Edges from origin vertex to tier 1 (origin -> teachers)
        """
        origin = self.mat_adj[0]
        copy = [0]
        copy = copy + subjects_offered.copy()

        for i in range(0, len(teachers)):
            destiny_teacher = i
            teacher_capacity = copy[i]
            self.addEdge(origin[i], destiny_teacher, teacher_capacity)

        self.mat_adj[0][0] = 0  # Removing link in origin vertex

    def setDestinyEdges(self, initial_vertex: int, subjects_info: list) -> None:
        """
        Set edges from subjects to destiny vertex

        :param initial_vertex: Vertex where starts tier 2 (Subjects)
        :param subjects_info: List of subjects in format [[Id, Name, Num_of_classes]]
        :return: Edges from each subject to destiny vertex (subject -> destiny)
        """
        subjects_capacities = [c[2] for c in subjects_info]
        destiny = self.num_vet - 1
        subject_capacity = None

        for i in range(initial_vertex, self.num_vet - 1):
            origin_subject = i
            for c in subjects_capacities:
                subject_capacity = c
                subjects_capacities.remove(c)
                break
            self.addEdge(origin_subject, destiny, subject_capacity)

    def setTeachersToSubjectsEdges(self):
        """
        Set edges from each teacher to their respective subjects

        :return: Updated graph data
        """
        teachersIndexes = self.teachers_index
        subjectsIndexes = self.subjects_index

        for key, (_, classes_offered, [*subjects]) in teachersIndexes.items():
            total_classes_offered = 0
            for subjectKey, (subjectId, _, classes) in subjectsIndexes.items():
                if total_classes_offered == len(subjects):
                    break
                if classes_offered == 0:
                    break
                if subjectId in subjects:
                    self.addEdge(key, subjectKey, classes)
                    total_classes_offered += 1

    def setInitialData(self, teachers_data: tuple, subjects_data: tuple):
        """
        Set the initial data of the graph

        :param teachers_data: Tuple in format (teachers: list, subjects_offered: list, subjects: list)
        :param subjects_data: Tuple in format (subjects_info: list, num_of_classes: list, total_of_subjects: int)
        :return: Updated graph data
        """
        (teachers, subjects_offered, subjects) = teachers_data
        (subjects_info, num_of_classes, total_of_subjects) = subjects_data

        # updating num_vet based on file data
        self.num_vet = 2 + len(teachers) + total_of_subjects

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
        self.setDestinyEdges(len(teachers) + 1, subjects_info)

        # setting key/value dictionary of teachers
        # and subjects in format {index: value}
        self.setTeachersAndSubjectsIndexes(len(teachers) + 1, subjects_info, teachers_data)

        # adding edge from each to teacher to respective
        # subject with capacity equals to subject number of classes
        self.setTeachersToSubjectsEdges()

        # updating num_edg after set all initial edges
        self.num_edg = len(teachers) + total_of_subjects + num_of_classes

    def bellmanFord(self, s: int, v: int) -> list:
        """
        Algorithm to get the shortest path from s to v

        :param s: origin vertex
        :param v: destiny vertex
        :return: shortest path
        """
        dist = [float("inf") for _ in range(len(self.list_adj))]
        pred = [None for _ in range(len(self.list_adj))]
        edges = self.getEdgesList()

        dist[s] = 0

        for i in range(0, len(self.list_adj) - 1):
            trade = False
            for source, destiny, [flow, capacity] in edges:  # edge = [source, destiny, (flow, capacity)]
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
        self.bellmanFord(1, 68)
