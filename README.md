# Sequence Assembly: Traveling Salesperson Problem (TSP) Model

In this problem, we will develop code for a toy version of an application we have seen in class: sequence assembly. We will look specifically at the Traveling Salesperson (TSP) model of the problem. We will assume that we are given a set of sequence fragments \(S_1, \ldots, S_n\) and wish to assemble them into a single longer sequence.

## a. TSP Graph Creation

We need to create a TSP graph. Provide pseudocode for a routine to transform a set of fragments into a graph with nodes corresponding to fragments and weighted edges between pairs of fragments weighted by the longest directed overlap between them. To simplify a bit, we will assume that they are all the same complementarity (so we do not need to consider reverse complements) and that our data is error-free (so we do not need to consider inexact overlaps).


## b. Greedy Heuristic

We were told that TSP is a hard problem so we probably will not have an exact solution. We will experiment with two solutions to the problem: a fast heuristic and an exact but exponential-time solution. First, write pseudocode for a greedy heuristic for the problem that works by repeatedly joining the two fragments with the highest overlap until all fragments have been joined into a single sequence.


## c. Branch-and-Bound Method

For an exact algorithm, we will use a simple branch-and-bound method of testing every possible ordering of fragments and then picking the one that maximizes the sum of overlaps. However, we will stop considering a partial ordering if it is already longer than our current best solution. We can initialize that with the greedy solution from part b as our initial bound. Provide pseudocode for this branch-and-bound method.


## d. Implementation

Write code implementing your methods. It should take as input a file with a set of fragment sequences, each on a separate line, e.g.,

ACTAGCATCGACTC GATCTAC CATTATCTCATCAGGCAT TCTTATTCT CTGCTTCATT


It should return the optimal order of fragments and the resulting sequence for each of the two methods, e.g.,

Greedy: 2-1-4-5-3 GATCTACTAGCATCGACTCTTATTCTGCTTCATTATCTCATCAGGCAT
Branch-and-bound: 2-1-4-5-3 GATCTACTAGCATCGACTCTTATTCTGCTTCATTATCTCATCAGGCAT


You can assume it should be able to work with any alphabet (so input could be nucleotides, amino acid sequences, binary sequences, etc.).

## e. Testing

Run your code on the provided test cases (`PS2test1.txt` and `PS2test2.txt`).
