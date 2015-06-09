import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt
import matplotlib
import collections
import scipy.signal as sci_sig
import operator
import copy
import networkx as nx
import csv
import glob
import os

def process_num(num_str):
    if num_str == "NA":
        return 0.0
    else:
        return float(num_str)

def open_vr(filename="vr_data.csv"):
    vr = []
    with open(filename, "rU") as vr_file:
        part_reader = csv.reader(vr_file)
        part_reader.next()
        for row in part_reader:
            vr.append(process_num(row[1]))
    return vr

def word_load(filename="corpus.txt"):
    with open(filename, "r") as corpus_file:
        words = corpus_file.read().split()
    curr_idx = 0
    word_map = {}
    word_idx = []
    for word in words:
        if word in word_map:
            word_idx.append(word_map[word])
        else:
            curr_idx += 1
            word_map[word] = curr_idx
            word_idx.append(curr_idx)
    return word_idx, word_map

def flip(dct):
    return {v:k for k,v in dct.iteritems()}

def logistic_map(x_n):
    r = 3.8 #chaos
    return r * x_n * (1 - x_n)

def discretize(data, bucket_size=0.2):
    data = np.array(data)
    data_min, data_max = np.min(data), np.max(data)
    buckets = np.arange(data_min, data_max, bucket_size)
    idx = np.digitize(data, buckets)
    return buckets[idx-1], idx

def stack_graph(data):
    graph = collections.defaultdict(list)
    for x, y in zip(data, data[1:]):
        graph[x].append(y)
    return graph

def traverse_stack_graph(graph, start):
    graph = copy.deepcopy(graph)
    curr_idx = start
    path = [start]
    while len(graph[curr_idx]) > 0:
        curr_member = graph[curr_idx].pop(0)
        curr_idx = curr_member
        path.append(curr_member)
    return path

def gen_logistic_map(n=1000):
    xs = []
    curr_x = 0.5
    for x in xrange(n):
        x_i = logistic_map(curr_x)
        curr_x = x_i
        xs.append(x_i)
    return np.array(xs)

def gen_disc_logistic_map(n=1000):
    return discretize(gen_logistic_map(n))

def path_to_words(path, flipped_map):
    words = []
    for word in path:
        words.append(flipped_map[word])
    return words

def test_logistic():
    logit = gen_logistic_map()
    logit_list = list(logit[1])
    logit_graph = stack_graph(logit_list)
    resampled_path = traverse_stack_graph(logit_graph, logit_list[1])[500:800]
    plt.close()
    plt.plot(logit_list[500:800], "b")
    plt.plot(resampled_path, "r")
    plt.show()

def logistic_degrees():
    logit = gen_logistic_map()
    logit_list = list(logit[1])
    logit_graph = stack_graph(logit_list)
    degrees = []
    for node, adjlist in logit_graph.items():
        degree = len(adjlist)
        degrees.append(degree)
    degrees = sorted(degrees, reverse=True)
    plt.loglog(degrees)
    plt.show()

def word_degrees():
    word_idx, word_map = word_load()
    word_graph = stack_graph(word_idx)
    degrees = []
    for node, adjlist in word_graph.items():
        degree = len(adjlist)
        degrees.append(degree)
    degrees = sorted(degrees, reverse=True)
    plt.loglog(degrees)
    plt.show()

def ts_degrees():
    def process_float(float_str):
        try:
            return float(float_str)
        except:
            return 0.0
    processed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*.csv_summed_*.csv")
    #unprocessed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*0.csv")
    curr_path = processed_globs[0]
    path_splits = os.path.split(curr_path)[1].split(".", 2)
    curr_fname = "".join([path_splits[0], path_splits[1]])
    print curr_fname
    with open(curr_path, "rU") as part_file:
        part_reader = csv.reader(part_file)
        curr_west = map(operator.itemgetter(1), part_reader)
        curr_west = curr_west[1:]
        curr_west = map(process_float, curr_west)
    _, ts_idx = discretize(curr_west)
    ts_graph = stack_graph(ts_idx)
    degrees = []
    for node, adjlist in ts_graph.items():
        degree = len(adjlist)
        degrees.append(degree)
    degrees = sorted(degrees, reverse=True)
    plt.loglog(degrees)
    plt.show()

def fbm_degrees(unstackify=False):
    fbm = np.load("quick_fbm.npy")
    _, fbm_idx = discretize(fbm)
    fbm_graph = stack_graph(fbm_idx)
    degrees = []
    for node, adjlist in fbm_graph.items():
        degree = len(adjlist)
        degrees.append(degree)
    degrees = sorted(degrees, reverse=True)
    plt.loglog(degrees)
    plt.show()

def test_fbm():
    pass

def test_words():
    idx = idx[:10]
    word_idx, word_map = word_load()
    word_graph = stack_graph(word_idx)
    flipped_map = flip(word_map)
    word_path = traverse_stack_graph(word_graph, word_idx[1])
    print " ".join(path_to_words(word_path[1000:5000], flipped_map))

    plt.plot(np.array(word_idx[:500]))
    plt.plot(np.array(word_path[:500]))
    f, pxx_den = sci_sig.periodogram(xs_new)
    f2, pxx_den2 = sci_sig.periodogram(xs)
    plt.semilogy(f, pxx_den)
    plt.semilogy(f2, pxx_den2)
    plt.show()

def spectrum_fbm():
    fbm = np.load("quick_fbm.npy")
    f, Pxx = sci_sig.periodogram(fbm)
    plt.close()
    plt.loglog(f, np.sqrt(Pxx))
    plt.xlabel("frequency")
    plt.ylabel("PSD")
    plt.savefig("spectrum_fbm")

def spectrum_logistic():
    logit = gen_logistic_map(1500)
    f, Pxx = sci_sig.periodogram(logit)
    plt.close()
    plt.loglog(f, np.sqrt(Pxx))
    plt.xlabel("frequency")
    plt.ylabel("PSD")
    plt.savefig("spectrum_logistic")

def spectrum_sinusoid():
    sinusoid = np.sin(np.linspace(0, 40 * np.pi, 1500))
    f, Pxx = sci_sig.periodogram(sinusoid)
    plt.close()
    plt.loglog(f, np.sqrt(Pxx))
    plt.xlabel("frequency")
    plt.ylabel("PSD")
    plt.savefig("spectrum_sinusoid")

def spectrum_vr():
    vr = open_vr()
    f, Pxx = sci_sig.periodogram(vr)
    plt.close()
    plt.loglog(f, np.sqrt(Pxx))
    plt.xlabel("frequency")
    plt.ylabel("PSD")
    plt.savefig("spectrum_vr")


def plot_fbm():
    fbm = np.load("quick_fbm.npy")[0:1500]
    plt.close()
    plt.plot(fbm)
    plt.xlabel("time")
    plt.ylabel("y")
    plt.savefig("plot_fbm")

def plot_logistic():
    logit = gen_logistic_map(1500)
    plt.close()
    plt.plot(logit)
    plt.xlabel("time")
    plt.ylabel("y")
    plt.savefig("plot_logistic")

def plot_sinusoid():
    sinusoid = np.sin(np.linspace(0, 40 * np.pi, 1500))
    plt.close()
    plt.xlabel("time")
    plt.ylabel("y")
    plt.plot(sinusoid)
    plt.savefig("plot_sinusoid")

def plot_vr():
    vr = open_vr()
    plt.close()
    plt.plot(vr)
    plt.xlabel("time")
    plt.ylabel("y")
    plt.savefig("plot_vr")

def graphify(ts):
    ts_map, idxs = discretize(ts, bucket_size=0.05)
    idxs = list(idxs)
    graph = stack_graph(idxs)
    resampled_path1 = traverse_stack_graph(graph, idxs[0])
    resampled_path2 = traverse_stack_graph(graph, idxs[1])
    return graph, resampled_path1, resampled_path2

def nx_graphify(ts):
    net = nx.DiGraph()
    graph, _, _ = graphify(ts)
    for tail, heads in graph.iteritems():
        for head in heads:
            net.add_edge(tail, head)
    return net

def plot_stack_deg(ts, name):
    graph, _, _ = graphify(ts)
    degrees = []
    for node, adjlist in graph.items():
        degree = len(adjlist)
        degrees.append(degree)
    degrees = sorted(degrees, reverse=True)
    plt.close()
    plt.xlabel("rank")
    plt.ylabel("degree")
    plt.loglog(degrees)
    plt.savefig(name)

def degree_fbm():
    fbm = np.load("quick_fbm.npy")[0:1500]
    plot_stack_deg(fbm, "degrees_fbm")

def degree_logistic():
    logit = gen_logistic_map(1500)
    plot_stack_deg(logit, "degrees_logit")

def degree_sinusoid():
    sinusoid = np.sin(np.linspace(0, 2 * np.pi, 1500))
    plot_stack_deg(sinusoid, "degrees_sinusoid")

def degree_vr():
    vr = open_vr()
    plot_stack_deg(vr, "degrees_vr")

def stats_fbm():
    fbm = np.load("quick_fbm.npy")[0:1500]
    net = nx_graphify(fbm)
    print "fbm path length: ", nx.average_shortest_path_length(net)
    print "fbm clustering: ", nx.average_clustering(net.to_undirected())

def stats_logistic():
    logit = gen_logistic_map(1500)
    net = nx_graphify(logit)
    print "logit path length: ", nx.average_shortest_path_length(net)
    print "logit clustering: ", nx.average_clustering(net.to_undirected())

def stats_sinusoid():
    sinusoid = np.sin(np.linspace(0, 40 * np.pi, 1500))
    net = nx_graphify(sinusoid)
    print "sinusoid path length: ", nx.average_shortest_path_length(net)
    print "sinusoid clustering: ", nx.average_clustering(net.to_undirected())

def stats_vr():
    vr = open_vr()
    net = nx_graphify(vr)
    print "vr path length: ", nx.average_shortest_path_length(net)
    print "vr clustering: ", nx.average_clustering(net.to_undirected())

def inverse_stack():
    vr = open_vr()
    _, resampled, _ = graphify(vr)
    plt.plot(resampled)
    plt.savefig("resampled_stack")

if __name__ == "__main__":
    font = {'size': 20}
    matplotlib.rc('font', **font)
    plot_fbm()
    plot_logistic()
    plot_sinusoid()
    plot_vr()
    degree_fbm()
    degree_logistic()
    degree_sinusoid()
    degree_vr()
    spectrum_fbm()
    spectrum_logistic()
    spectrum_sinusoid()
    spectrum_vr()
    inverse_stack()
