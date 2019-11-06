# Calculate metrics for relations in conceptnet

from collections import defaultdict
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet


def is_in_category(categories, syn, depth=0) -> bool:
    if syn.name() in cached:
        return cached[syn.name()]
    else:
        result = False
        if depth > 15:
            return result
        elif syn.name().split('.')[0] in categories:
            result = True
        else:
            result = any(is_in_category(categories, hypernym, depth=depth + 1)
                         for hypernym in syn.hypernyms())
        cached[syn.name()] = result
        return result


categories = [('physical', ['object', 'physical_object', 'physical_entity']), ('person', ['person'])]

for category in categories:
    relations = defaultdict(int)
    relation_fn_counts = defaultdict(int)
    relation_sn_counts = defaultdict(int)
    relation_fn_not_counts = defaultdict(int)
    relation_sn_not_counts = defaultdict(int)
    relation_physical_fn_counts = defaultdict(int)
    relation_physical_sn_counts = defaultdict(int)
    wn_lemmas = set(wordnet.all_lemma_names())
    first_node_lengths = defaultdict(list)
    second_node_lengths = defaultdict(list)

    cached = {}
    file = 'datasets/relations-with-categories.txt'
    # Iterate over, get what percent of the notes connected by the relations are physical
    with open(file, 'r') as inf:
        for line in inf:
            relation, _, node_1, _, node_2 = line.strip().split(', ')

            relations[relation] += 1
            x_is_in_category, y_is_in_category = False, False

            # increment counters if the node is in the category
            if node_1 in wn_lemmas:
                relation_fn_counts[relation] += 1
                if any(is_in_category(category[1], syn) for syn in wordnet.synsets(node_1)):
                    relation_physical_fn_counts[relation] += 1
                    x_is_in_category = True
            else:
                relation_fn_not_counts[relation] += 1
                first_node_lengths[relation].append(len(node_1.split('_')))
            if node_2 in wn_lemmas:
                relation_sn_counts[relation] += 1
                if any(is_in_category(category[1], syn) for syn in wordnet.synsets(node_2)):
                    relation_physical_sn_counts[relation] += 1
                    y_is_in_category = True
            else:
                relation_sn_not_counts[relation] += 1
                second_node_lengths[relation].append(len(node_2.split('_')))

    # Print to output file
    with open('datasets/' + category[0] + '-relations-metrics.txt', 'w') as out_file:
        print('Relation, count, x-Ratio, y-Ratio, x-OOV, y-OOV, x-AveOOVLength, y-AveOOVLength', file=out_file)
        for relation, count in relations.items():
            r1c = relation_fn_counts[relation]
            r1c_not = relation_fn_not_counts[relation]
            r2c = relation_sn_counts[relation]
            r2c_not = relation_sn_not_counts[relation]
            n1c = relation_physical_fn_counts[relation]
            n2c = relation_physical_sn_counts[relation]
            lengths1 = first_node_lengths[relation]
            lengths2 = second_node_lengths[relation]
            print(
                '%s, %d, %1.3f, %1.3f, %1.3f, %1.2f, %1.3f, %1.2f' % (relation, count, n1c / (r1c + 1), n2c / (r2c + 1),
                                                                      r1c_not / (r1c + r1c_not + 1),
                                                                      r2c_not / (r2c + r2c_not + 1),
                                                                      sum(lengths1) / (len(lengths1) + 1),
                                                                      sum(lengths2) / (len(lengths2) + 1)),
                file=out_file)
