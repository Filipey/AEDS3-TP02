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
        self.num_of_classes = None
        self.edges_list = []

    def addEdge(self, source, sink, capacity=float("inf"), flow=0) -> None:
        """
        Add an edge on graph in format (source, sink, (flow, capacity))

        :param flow: Flow value
        :param source: Source vertex
        :param sink: Destiny vertex
        :param capacity: Capacity of the edge
        """
        if source < self.num_vet and sink < self.num_vet:
            self.mat_adj[source][sink] = [flow, capacity]
            self.list_adj[source].append((sink, [flow, capacity]))
            self.num_edg += 1
        else:
            sys.exit("Invalid Edge")

    def removeEdge(self, source, sink) -> None:
        """
        Delete a edge from graph

        :param source: Source vertex
        :param sink: Destiny vertex
        """
        if source < self.num_vet and sink < self.num_vet:
            if self.mat_adj[source][sink] != 0:
                self.num_edg -= 1
                self.mat_adj[source][sink] = 0

                for (v, w) in self.list_adj[source]:
                    if v == sink:
                        self.list_adj[source].remove((v, w))
                        break

                self.num_edg += 1
        else:
            sys.exit("Invalid Edge")

    def SetEdgesList(self) -> None:
        """
        Set list of edges in format: source_vertex, sink_vertex, (flow, capacity)

        :return: Updated list of edges
        """

        for i in range(0, len(self.mat_adj)):
            for j in range(0, len(self.mat_adj[i])):
                if self.mat_adj[i][j] != 0:
                    if type(self.mat_adj[i][j]) is list:
                        [flow, _] = self.mat_adj[i][j]
                    else:
                        flow = self.mat_adj[i][j]
                    self.edges_list.append((i, j, flow))

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

    def readSubjects(self, filename: str) -> tuple:
        """
        Read subjects file and return formatted data

        :param filename: Name of the subjects csv file in /dataset
        :return: subjects_info: list, num_of_classes: int, total_of_subjects: list
        """
        try:
            df = pd.read_csv("dataset/" + filename, sep=";")

            data = df.iloc[:, ].values.tolist()  # Get all data into a list

            num_of_classes = data.pop(-1)[2]  # Remove last line that contains the total of classes

            self.num_of_classes = num_of_classes

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

        for i in range(0, len(teachers)):
            self.teachers_index[i+1] = (teachers[i], num_of_subject_offered[i], subjects_offered[i])

        for j in range(subjects_initial_vertex, self.num_vet - 1):
            for subject in subjects_info:
                self.subjects_index[j] = subject
                subjects_info.remove(subject)
                break

    def setSourceEdges(self, teachers: list, subjects_offered: list) -> None:
        """
        Set edges from source vertex to teachers

        :param teachers: List of teachers
        :param subjects_offered: List of subjects that each teacher apply
        :return: Edges from source vertex to tier 1 (source -> teachers)
        """
        source = self.mat_adj[0]
        copy = [0]
        copy = copy + subjects_offered.copy()

        for i in range(0, len(teachers) + 1):
            sink_teacher = i
            teacher_capacity = copy[i]
            self.addEdge(source[i], sink_teacher, teacher_capacity)

        # Removing link in source vertex
        self.mat_adj[0][0] = 0
        self.list_adj[0].pop(0)

    def setSinkEdges(self, initial_vertex: int, subjects_info: list) -> None:
        """
        Set edges from subjects to sink vertex

        :param initial_vertex: Vertex where starts tier 2 (Subjects)
        :param subjects_info: List of subjects in format [[Id, Name, Num_of_classes]]
        :return: Edges from each subject to sink vertex (subject -> sink)
        """
        subjects_capacities = [c[2] for c in subjects_info]
        sink = self.num_vet - 1
        subject_capacity = None

        for i in range(initial_vertex, self.num_vet - 1):
            source_subject = i
            for c in subjects_capacities:
                subject_capacity = c
                subjects_capacities.remove(c)
                break
            self.addEdge(source_subject, sink, subject_capacity)

    def setTeachersToSubjectsEdges(self):
        """
        Set edges from each teacher to their respective subjects

        :return: Updated graph data
        """
        teachersIndexes = self.teachers_index
        subjectsIndexes = self.subjects_index

        # flow value based on the preferences table
        flow = [0, 3, 5, 8, 10]

        for key, (_, classes_offered, [*subjects]) in teachersIndexes.items():
            total_classes_offered = 0
            for subjectKey, (subjectId, _, classes) in subjectsIndexes.items():
                if total_classes_offered == len(subjects):
                    break
                if classes_offered == 0:
                    break
                if subjectId in subjects:
                    self.addEdge(key, subjectKey, classes, flow[subjects.index(subjectId)])
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

        # adding edge from source 's' to each teacher
        # with capacity equals to their subjects_offered
        self.setSourceEdges(teachers, subjects_offered)

        # adding edge from each subject to sink 't'
        # with capacity equals to number of classes of the subject
        self.setSinkEdges(len(teachers) + 1, subjects_info)

        # setting key/value dictionary of teachers
        # and subjects in format {index: value}
        self.setTeachersAndSubjectsIndexes(len(teachers) + 1, subjects_info, teachers_data)

        # adding edge from each to teacher to respective
        # subject with capacity equals to subject number of classes
        self.setTeachersToSubjectsEdges()

        self.SetEdgesList()

    def bellmanFord(self, s: int, v: int) -> list:
        """
        Algorithm to get the shortest path from s to v

        :param s: source vertex
        :param v: sink vertex
        :return: shortest path
        """
        dist = [float("inf") for _ in range(len(self.list_adj))]
        pred = [None for _ in range(len(self.list_adj))]
        edges = self.edges_list

        dist[s] = 0

        for i in range(0, len(self.list_adj) - 1):
            trade = False
            for source, sink, flow in edges:  # edge = [source, sink, flow]
                if dist[sink] > dist[source] + flow:
                    dist[sink] = dist[source] + flow
                    pred[sink] = source
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

        if len(shortest_path) == 1:
            shortest_path.clear()

        shortest_path.reverse()

        return shortest_path

    def getFlowByVertex(self):
        """
        Get the flow that should pass for each vertex

        :return: List with flow of each vertex
        """
        b = [self.num_of_classes]

        for _, [_, flow, [*_]] in self.teachers_index.items():
            b.append(flow)

        for _, [_, _, flow] in self.subjects_index.items():
            b.append(flow)

        b.append(-self.num_of_classes)

        return b

    def getFlowAndCapacityOfEachEdge(self) -> tuple:
        """
        Get the flow passed of each edge in the graph

        :return: Matrix with flow of each edge
        """

        flow_of_edge = [[0 for _ in range(len(self.mat_adj))] for _ in range(len(self.mat_adj))]
        capacity_of_edges = [[0 for _ in range(len(self.mat_adj))] for _ in range(len(self.mat_adj))]

        for i in range(0, len(self.mat_adj)):
            for j in range(0, len(self.mat_adj[i])):
                if self.mat_adj[i][j] != 0:
                    [flow, capacity] = self.mat_adj[i][j]
                    flow_of_edge[i][j] = flow
                    capacity_of_edges[i][j] = capacity

        return flow_of_edge, capacity_of_edges

    def successfulShortestPaths(self, s: int, t: int):
        F = [[0 for _ in range(len(self.mat_adj))] for _ in range(len(self.mat_adj))]
        flow_for_vertex = self.getFlowByVertex()
        flow_of_edges, capacity_of_edges = self.getFlowAndCapacityOfEachEdge()
        shortest_path = self.bellmanFord(s, t)

        while len(shortest_path) != 0 and flow_for_vertex[s] != 0:
            f = float("inf")
            for i in range(1, len(shortest_path)):
                u = shortest_path[i - 1]
                v = shortest_path[i]

                if capacity_of_edges[u][v] < f:
                    f = capacity_of_edges[u][v]

            for i in range(1, len(shortest_path)):
                u = shortest_path[i - 1]
                v = shortest_path[i]
                F[u][v] += f
                capacity_of_edges[u][v] -= f
                capacity_of_edges[v][u] += f
                flow_for_vertex[s] -= f
                flow_for_vertex[t] += f

                if capacity_of_edges[u][v] == 0:
                    self.mat_adj[u][v] = 0
                    self.edges_list.remove((u, v, flow_of_edges[u][v]))

                if self.mat_adj[v][u] == 0:
                    self.mat_adj[v][u] = 1
                    self.edges_list.append((v, u, -flow_of_edges[u][v]))
                    flow_of_edges[v][u] = -flow_of_edges[u][v]

            shortest_path = self.bellmanFord(s, t)

        return F

    def run(self, teachers_file: str, subjects_file: str) -> None:
        self.setInitialData(self.readTeachers(teachers_file), self.readSubjects(subjects_file))
        self.successfulShortestPaths(0, self.num_vet - 1)
