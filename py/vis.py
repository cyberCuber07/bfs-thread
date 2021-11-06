
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


def load_data(path):
    f = csv.reader(open(path, "r"))

    data = np.array([[int(float(x)) for x in line] for line in f])

    return data


class Graph:
    def __init__(self, path):
        self.data = load_data(path)
        self.x, self.y, self.w = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        self.adj = self.edge2adj()
        self.vis = np.zeros((self.data.shape[0]), dtype=bool)
        self.graphs = self.get_graphs()

    def get_max_num(self):
        max_val = 0
        any = 0
        for one in self.data:
            max_val = max(max_val, one[0])
            max_val = max(max_val, one[1])
            if one[0] * one[1] == 0:
                any = 1
        return int(max_val + any)

    def edge2adj(self):
        max_val = self.get_max_num()
        adj = [[] for _ in range(max_val)]
        for e in self.data:
            adj[e[0]].append(e)
        return adj

    def one_graph(self, e):
        q = [e]
        graph = []
        while len(q):
            tmp = q[-1]
            src = tmp[0]
            q.pop(-1)
            if not self.vis[src]:
                self.vis[src] = True
                graph.append(tmp)
                for e in self.adj[src]:
                    q.insert(0, e)
        print(graph)
        return graph

    def get_graphs(self):
        graphs = []
        for _x, _y, _w in zip(self.x, self.y, self.w):
            graph = self.one_graph([_x, _y, _w])
            if len(graph) > 0:
                graphs.append(np.array(graph))
        return graphs
    
    def visualize(self):
        for graph in self.graphs:
            x_tmp, y_tmp = graph[:, 0], graph[:, 1]
            for i in range( x_tmp.shape[0] // 2 ):
                idx = 2 * i
                _x = [x_tmp[idx], x_tmp[idx + 1]]
                _y = [y_tmp[idx], y_tmp[idx + 1]]
                plt.plot( _x, _y, 'ro-' )
            # plt.scatter(x, y, linewidths=1)
            plt.show()


def main():
    path = sys.argv[1]
    g = Graph(path)
    g.visualize()


if __name__ == "__main__":
    main()

