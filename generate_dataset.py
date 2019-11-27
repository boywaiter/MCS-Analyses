import collections
import os
from collections import defaultdict, Counter
from copy import copy
from typing import List

from networkx import DiGraph, isolates
import pickle
import random as r
from nltk.corpus import wordnet

wn_lemmas = set(wordnet.all_lemma_names())

# commonsense_relations = ['UsedFor', 'AtLocation', 'HasSubevent', 'HasPrerequisite', 'CapableOf', 'Causes', 'PartOf',
#                          'MannerOf', 'MotivatedByGoal', 'HasProperty', 'ReceivesAction', 'HasA', 'CausesDesire',
#                          'HasFirstSubevent', 'Desires', 'NotDesires', 'HasLastSubevent', 'MadeOf', 'NotCapableOf',
#                          'CreatedBy', 'LocatedNear']
commonsense_relations = ['UsedFor', 'AtLocation', 'HasSubevent', 'HasPrerequisite', 'CapableOf', 'Causes', 'PartOf',
                         'MannerOf', 'MotivatedByGoal', 'HasProperty', 'ReceivesAction', 'HasA', 'CausesDesire',
                         'HasFirstSubevent', 'Desires', 'HasLastSubevent', 'MadeOf', 'LocatedNear']

relation_to_string = {rel: ''.join(c if c.islower() else ' ' + c for c in rel).lower()[1:]
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


def dfs_helper(graph, from_node, visited, depth, k):
    visited[from_node] = True
    if depth == k:
        return [from_node]
    neighbors = list(graph.neighbors(from_node))
    r.shuffle(neighbors)
    for next_node in neighbors:
        if not visited[next_node]:
            recursion_result = dfs_helper(graph, next_node, visited, depth + 1, k)
            if len(recursion_result) == 1 and not isinstance(recursion_result[0], tuple):
                return [(from_node, graph.get_edge_data(from_node, next_node)['label'], next_node)]
            else:
                return [(from_node, graph.get_edge_data(from_node, next_node)['label'], next_node)] + recursion_result
    return [from_node]


def generate_khop(graph: DiGraph, *, k=2, number_of_samples=10) -> List[List[tuple]]:
    print('Generating k-hops...')
    paths = []
    nodes_to_sample = copy(graph.nodes)
    # Find new k-hop paths until we had enough or no more root nodes
    while len(paths) < number_of_samples and len(graph.nodes) > 0:
        if number_of_samples > len(nodes_to_sample):
            samples = nodes_to_sample
        else:
            samples = r.sample(nodes_to_sample, number_of_samples)
        for root in samples:
            # for each node, get their first neighbor, and recursively form a path
            visited = {node: False for node in graph.nodes}
            path = dfs_helper(graph, root, visited, depth=0, k=k)
            if len(path) == k:
                paths.append(path)
        nodes_to_sample = [node for node in nodes_to_sample if node not in samples]

    paths = paths[:min(number_of_samples, len(paths))]
    return paths


def filter_non_commonsense(graph: DiGraph, commonsense_relations: List[str]) -> DiGraph:
    print('Filtering commonsense relations...')
    # Remove edges that are not listed as commonsense
    edges_to_remove = []
    for x, y in graph.edges:
        data = graph.get_edge_data(x, y)
        if data['label'] not in commonsense_relations or data['weight'] == -1:
            edges_to_remove.append((x, y))
    new_graph = copy(graph)
    new_graph.remove_edges_from(edges_to_remove)
    # Remove possible leftover island nodes
    islands = list(isolates(new_graph))
    new_graph.remove_nodes_from(islands)
    return new_graph


def generate_cs_datasets(relation_data, n_samples=100000):
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
        physical_data.extend(r.sample(all_data, n_samples - len(physical_data)))
    if len(person_data) > n_samples:
        person_data = r.sample(person_data, n_samples)
    else:
        person_data.extend(r.sample(all_data, n_samples - len(person_data)))

    pairs = [('random-cs-n' + str(n_samples), all_data),
             ('physical-cs-n' + str(n_samples), physical_data),
             ('person-cs-n' + str(n_samples), person_data)]
    for name, data in pairs:
        with open('datasets/' + name + '.txt', 'w') as outf:
            for data_point in data:
                print(data_point, file=outf)


def generate_dataset_from_tuple_lists(file_name, tuple_lists):
    list_as_strings = []
    for list_of_tuples in tuple_lists:
        words = []
        for x, r, y in list_of_tuples:
            words += ['and', x, relation_to_string[r], y]
        list_as_strings.append(' '.join(words[1:]))

    with open('datasets/' + file_name + '.txt', 'w') as outf:
        for sentence in list_as_strings:
            print(sentence, file=outf)


def generate_all_cs_dataset(filtered_graph: DiGraph):
    relations = []
    for node in filtered_graph.nodes():
        for neighbor in filtered_graph.neighbors(node):
            relation = filtered_graph.get_edge_data(node, neighbor)['label']
            rs = ''.join(c if c.islower() else ' ' + c for c in relation).lower()[1:]
            relations.append(node + ' ' + rs + ' ' + neighbor)
    with open('datasets/all_cs_217k.txt', 'w') as outfile:
        for relation in relations:
            print(relation, file=outfile)


def generate_multiple_choice_dataset(choice_matrix_path: str, filtered_graph: DiGraph, n_choices=7):
    incorrect_choices_dict = {}
    with open(choice_matrix_path, 'r') as matrix_file:
        for line in matrix_file.readlines()[1:]:
            tokens = line.split(',')
            incorrect_choices_dict[tokens[0]] = [token for token in tokens[1:] if token != '' and token != '\n' and
                                                 token in commonsense_relations]
    question_tuples = []
    id_number = 0
    counter = collections.Counter()
    for e1 in filtered_graph.nodes():
        for e2 in filtered_graph.neighbors(e1):
            counter.update([(e1,e2)])
            correct_relation = filtered_graph.get_edge_data(e1, e2)['label']
            incorrect_relations = incorrect_choices_dict[correct_relation]
            question_tuples.append((id_number, e1, e2, correct_relation, incorrect_relations))
            id_number += 1
    if max(counter.values()) > 1:
        raise RuntimeError('Some pair of entities appear more than once.')
    r.shuffle(question_tuples)

    with open('datasets/cn-all-cs-multiple-choice-data.jsonl', 'w') as data_file:
        with open('datasets/cn-all-cs-multiple-choice-labels.lst', 'w') as labels_file:
            for id_number, e1, e2, correct_relation, incorrect_relations in question_tuples:
                final_choices: List = r.sample(incorrect_relations, n_choices - 1) + [correct_relation]
                r.shuffle(final_choices)
                correct_index = final_choices.index(correct_relation)
                choice_strings = ['\"sol' + str(i + 1) + '\": \"' + relation_to_string[final_choices[i]] + '\"' for i in
                                  range(n_choices)]
                line_to_print = '{\"id\": \"' + str(id_number) + '\", \"e1\": \"' + e1.replace('_',' ') + \
                                '\", \"e2\": \"' + e2.replace('_',' ') + '\", ' + ', '.join(choice_strings) +'}'
                print(line_to_print, file=data_file)
                print(str(correct_index), file=labels_file)


if __name__ == '__main__':
    r.seed(0)
    file_path = 'datasets/relations-with-categories.txt'
    pickle_file_path = 'datasets/conceptnet_digraph.pickle'
    choice_matrix_path = 'datasets/CN-Relation-ClassificationMatrix.csv'
    conceptnet = load_graph_file(file_path, pickle_file_path)

    # Filter out non wordnet nodes
    # wn_conceptnet = filter_non_wordnet(conceptnet)
    # print(len(wn_conceptnet.nodes))
    # print(len(wn_conceptnet.edges))

    # Filter out non commonsense edges
    filtered_graph = filter_non_commonsense(conceptnet, commonsense_relations)

    # Generate K-hop dataset
    # k = 2
    # n_samples=100
    # khops = generate_khop(filtered_graph, k=k, number_of_samples=n_samples)
    # generate_dataset_from_tuple_lists('khops_k='+str(k)+'_n='+str(n_samples), khops)

    # Generate 100k CS datasets for testing
    # commonsense_relation_data = []
    # with open(file_path, 'r') as infile:
    #     for line in infile:
    #         tokens = line.strip().split(', ')
    #         if tokens[0] in commonsense_relations:
    #             commonsense_relation_data.append(tokens)
    #
    # generate_cs_datasets(commonsense_relation_data)

    generate_multiple_choice_dataset(choice_matrix_path, filtered_graph)
