import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt
import collections

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

def logit_map():
    xs = []
    curr_x = 0.5
    for x in xrange(1000):
        x_i = logistic_map(curr_x)
        curr_x = x_i
        xs.append(x_i)
    xs = np.array(xs)
    return discretize(xs)

def path_to_words(path, flipped_map):
    words = []
    for word in path:
        words.append(flipped_map[word])
    return words


if __name__ == "__main__":
    #idx = idx[:10]
    word_idx, word_map = word_load()
    word_graph = stack_graph(word_idx)
    flipped_map = flip(word_map)
    word_path = traverse_stack_graph(word_graph, word_idx[1])
    print path_to_words(word_path[1000:1100], flipped_map)

    #plt.plot(np.array(word_idx[:500]))
    #plt.plot(np.array(word_path[:500]))
    #f, pxx_den = sci_sig.periodogram(xs_new)
    #f2, pxx_den2 = sci_sig.periodogram(xs)
    #plt.semilogy(f, pxx_den)
    #plt.semilogy(f2, pxx_den2)
    #plt.show()
