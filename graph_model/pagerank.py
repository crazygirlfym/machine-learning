# --*- coding:utf-8 -**

import math
import random
import networkx as nx 
import csv

class PageRank:
    def __init__(self, graph, max_iterations):
        self.graph = graph 
        self.V = len(self.graph.nodes())
        self.d = 0.85
        self.ranks = dict()
        self.max_iterations = max_iterations  
        self.min_delta = 0.00001    
        self.init_rank()    
    def init_rank(self):
        for node in self.graph.nodes():
            self.ranks[node] = 1.0 / self.V 


    def rank(self):

        for i in range (self.max_iterations):
            change = 0 
            for key in self.graph.nodes():
                rank_sum = 0
                curr_rank = self.ranks[key]

                neighbors = self.graph.out_edges(key)
                for n in neighbors:
                    outlinks = len(self.graph.out_edges(n[1]))
                    if outlinks > 0:
                        rank_sum += (1 / float(outlinks)) * self.ranks[n[1]]
                ## 迭代公式
                rank_tmp = (1 - float(self.d)) * (1/float(self.V)) + self.d * rank_sum 
                change += abs(rank_tmp - self.ranks[key])
                
                self.ranks[key] = rank_tmp 
            if change < self.min_delta:
                print ("finished in %s iterations!" %i)
                return 


class Graph:
    
    def __init__(self, filename):
        self.filename = filename 
        self.data = self.readData(self.filename)
        self.graph = self.buildGraph() 
    def readData(self, filename):
        reader = csv.reader(open(filename, 'r'), delimiter=',')
        data = [row for row in reader]
        return data


    def buildGraph(self):
        DG = nx.DiGraph()

        for i, row in enumerate(self.data):
            node_a = row[0].split("\t")[0]
            node_b = row[0].split("\t")[1]
            
            DG.add_edge(node_a, node_b)
        return DG 

if __name__ == "__main__":
    g = Graph("./graph.txt")
    p = PageRank(g.graph, 100)
    p.rank()
    print (g.graph.out_edges("A"))
    print (g.graph.nodes())
