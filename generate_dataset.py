import os
from collections import defaultdict
from copy import copy
from typing import List

from networkx import DiGraph, isolates
import pickle
import random as r
from nltk.corpus import wordnet
wn_lemmas = set(wordnet.all_lemma_names())

commonsense_relations = ['UsedFor', 'AtLocation', 'HasSubevent', 'HasPrerequisite', 'CapableOf', 'Causes', 'PartOf',
                         'MannerOf', 'MotivatedByGoal', 'HasProperty', 'ReceivesAction', 'HasA', 'CausesDesire',
                         'HasFirstSubevent', 'Desires', 'NotDesires', 'HasLastSubevent', 'MadeOf', 'NotCapableOf',
                         'CreatedBy', 'LocatedNear']

relation_to_string = {rel: ''.join(c if c.islower() else ' '+c for c in rel).lower()[1:]
                      for rel in commonsense_relations}


def graph_dict_to_digraph(graph_dict):
    # Helper function to convert a graph dictionary into a digraph
    conceptnet_graph = DiGraph()
    for x, tuples in graph_dict.items():
        conceptnet_graph.add_node(x)
        for relation, y in tuples:
            conceptnet_graph.add_node(y)
            conceptnet_graph.add_edge(x, y, label=relation, weight=1)
            conceptnet_graph.add_edge(y, x, label=relation, weight=-1)
    return conceptnet_graph


def load_graph_file(path, pickle_file) -> DiGraph:
    # load if there is a pickled version, if not parse it
    if os.path.exists(pickle_file):
        print('Loading pickled file...')
        with open(pickle_file, 'rb') as infile:
            pickled_data = pickle.load(infile)
            return pickled_data

    print('Parsing graph from relations csv...')
    # otherwise parse from scratch
    graph_as_dict = defaultdict(list)
    with open(path, 'r') as infile:
        for line in infile:
            relation, _, x, _, y = line.strip().split(', ')
            graph_as_dict[x].append((relation, y))

    conceptnet_digraph = graph_dict_to_digraph(graph_as_dict)
    with open(pickle_file, 'wb') as outfile:
        pickle.dump(conceptnet_digraph, outfile, pickle.HIGHEST_PROTOCOL)
    return conceptnet_digraph


def filter_non_wordnet(graph: DiGraph) -> DiGraph:
    # Returns a new graph with non wn nodes removed
    nodes = graph.nodes
    nodes_to_remove = []
    # if any part of the node is not the wordnet, remove it from graph
    for node in nodes:
        if not all([token in wn_lemmas for token in node.split()]):
            nodes_to_remove.append(node)
    new_graph = copy(graph)
    new_graph.remove_nodes_from(nodes_to_remove)
    return new_graph


def generate_khop(graph: DiGraph, *, k=2, number_of_samples=10) -> List[List[tuple]]:
    paths = []
    # Find new k-hop paths until we had enough
    while len(paths) < number_of_samples:
        root = r.sample(graph.nodes, 1)[0]
        # for each node, get their first neighbor, and recursively form a path
        node = root
        path = []
        for i in range(k):
            neighbors = [n for n in graph.neighbors(node) if graph.get_edge_data(node, n)['weight'] != -1]
            if neighbors:
                next_node = neighbors[0]
                path.append((node, next_node, graph.get_edge_data(node, next_node)))
                node = next_node
            else:
                break
        if len(path) == k:
            paths.append(path)

    with open('datasets/khops_k='+str(k)+'_samples='+str(number_of_samples)+'.txt', 'w') as outfile:
        for path in paths:
            string = '; '.join([x+' '+data['label']+' '+y for x, y, data in path])
            print(string, file=outfile)


def filter_non_commonsense(graph: DiGraph, commonsense_relations: List[str]) -> DiGraph:
    # Remove edges that are not listed as commonsense
    edges_to_remove = []
    for x, y in graph.edges:
        data = graph.get_edge_data(x, y)
        if data['label'] not in commonsense_relations or data['weight'] == -1:
            edges_to_remove.append((x,y))
    new_graph = copy(graph)
    new_graph.remove_edges_from(edges_to_remove)
    # Remove possible leftover island nodes
    islands = list(isolates(new_graph))
    new_graph.remove_nodes_from(islands)
    return new_graph


def generate_datasets(relation_data, n_samples=100000):
    person_data, physical_data, all_data = [], [], []
    for relation, x_category, x, y_category, y in relation_data:
        # convert data to string data point
        x_string = ' '.join(x.split('_'))
        y_string = ' '.join(y.split('_'))
        relation_string = relation_to_string[relation]
        data_point = x_string + ' ' + relation_string + ' ' + y_string

        # create each data set category, random, person data, physical data
        all_data.append(data_point)
        if x_category == 'PER' or y_category == 'PER':
            person_data.append(data_point)
            physical_data.append(data_point)
        if x_category == 'PHYS' or y_category == 'PHYS':
            physical_data.append(data_point)

    # sample n_sampels for each data group
    if len(all_data) > n_samples:
        all_data = r.sample(all_data, n_samples)
    if len(physical_data) > n_samples:
        physical_data = r.sample(physical_data, n_samples)
    else:
        physical_data.extend(r.sample(all_data, n_samples-len(physical_data)))
    if len(person_data) > n_samples:
        person_data = r.sample(person_data, n_samples)
    else:
        person_data.extend(r.sample(all_data, n_samples-len(person_data)))

    pairs = [('random-cs-n'+str(n_samples), all_data),
             ('physical-cs-n'+str(n_samples), physical_data),
             ('person-cs-n'+str(n_samples), person_data)]
    for name, data in pairs:
        with open('datasets/'+name+'.txt', 'w') as outf:
            for data_point in data:
                print(data_point, file=outf)


if __name__ == '__main__':
    r.seed(0)
    file_path = 'datasets/relations-with-categories.txt'
    pickle_file_path = 'datasets/conceptnet_digraph.pickle'
    # conceptnet = load_graph_file(file_path, pickle_file_path)

    # Filter out non wordnet nodes
    # wn_conceptnet = filter_non_wordnet(conceptnet)
    # print(len(wn_conceptnet.nodes))
    # print(len(wn_conceptnet.edges))

    # Filter out non commonsense edges
    # filtered_graph = filter_non_commonsense(conceptnet, commonsense_relations)
    # k = 2
    # n_samples=100
    # khops = generate_khop(filtered_graph, k=k, number_of_samples=n_samples)

    # Generate dataset for testing
    commonsense_relation_data = []
    with open(file_path, 'r') as infile:
        for line in infile:
            tokens = line.strip().split(', ')
            if tokens[0] in commonsense_relations:
                commonsense_relation_data.append(tokens)

    generate_datasets(commonsense_relation_data)

