import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt
import collections

def logistic_map(x_n):
    r = 3.8 #chaos
    return r * x_n * (1 - x_n)

def discretize(data, bucket_size=0.01):
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
    curr_idx = start
    path = [start]
    while len(graph[curr_idx]) > 0:
        curr_member = graph[curr_idx].pop(0)
        curr_idx = curr_member
        path.append(curr_member)
    return path

if __name__ == "__main__":
    xs = []
    curr_x = 0.5
    for x in xrange(1000):
        x_i = logistic_map(curr_x)
        curr_x = x_i
        xs.append(x_i)
    xs = np.array(xs)
    xs, idx = discretize(xs)
    #idx = idx[:10]
    list_idx = list(idx)
    graph = stack_graph(idx)
    path = traverse_stack_graph(graph, idx[1])
    xs_new = xs[path]

    plt.plot(xs_new)
    plt.plot(xs)
    #f, pxx_den = sci_sig.periodogram(xs_new)
    #f2, pxx_den2 = sci_sig.periodogram(xs)
    #plt.semilogy(f, pxx_den)
    #plt.semilogy(f2, pxx_den2)
    plt.show()
