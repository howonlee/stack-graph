Reversible dual from graph to discrete-valued time series, hopefully

Stack graph

Imagine a time series that goes like this: 1, 4, 6, 2, whatever

Each node is obviously one of the values: 1 is a node, 4, etc

But each node also keeps track of a stack of "nexts", which pops out whenever we are at its location. They can repeat, meaning it's not an adjacency list. But an adjacency list would be a stack graph

This is a reversible and deterministic dual from graphs to time series and vice versa, or at least I claim
