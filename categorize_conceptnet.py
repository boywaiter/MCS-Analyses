# Script to add categories to relations from conceptnet

from collections import defaultdict

from nltk.corpus import wordnet

physical_relations = defaultdict(int)
person_relations = defaultdict(int)
not_in_wn_relations = defaultdict(int)
wn_lemmas = set(wordnet.all_lemma_names())

# Categories and their wordnet hypernym descriptions
categories = ['PER', 'PHYS']
category_to_descriptions = {
    'PHYS': ['object', 'physical_object', 'physical_entity'],
    'PER': ['person']
}

cached = {category: {} for category in categories}
relations = defaultdict(list)


def is_in_category(category, syn, depth=0) -> bool:
    if syn.name() in cached[category]:
        return cached[category][syn.name()]
    else:
        result = False
        if depth > 15:
            return result
        elif syn.name().split('.')[0] in category_to_descriptions[category]:
            result = True
        else:
            result = any(is_in_category(category, hypernym, depth=depth + 1)
                         for hypernym in syn.hypernyms())
        cached[category][syn.name()] = result
        return result


# Iterate over, get what percent of the notes connected by the relations are physical
with open('datasets/simplified_english_conceptnet.csv', 'r') as inf:
    for line in inf:
        node_1, relation, node_2 = line.split()[0:3]

        tokens = [relation]
        for node in [node_1, node_2]:
            label = 'NOT_WN'
            if len(node.split('_')) < 2:
                new_node = ''
                if node not in wn_lemmas:
                    nouns = wordnet._morphy(node, wordnet.NOUN)
                    if len(nouns) > 0:
                        new_node = nouns[0]
                    else:
                        verbs = wordnet._morphy(node, wordnet.VERB)
                        if len(verbs) > 0:
                            new_node = verbs[0]
                        else:
                            adjs = wordnet._morphy(node, wordnet.ADJ)
                            if len(adjs) > 0:
                                new_node = adjs[0]
                    if new_node in wn_lemmas:
                        node = new_node

                if node in wn_lemmas:
                    in_category = False
                    for category in categories:
                        if any(is_in_category(category, syn) for syn in wordnet.synsets(node)):
                            in_category = True
                            label = category
                            break
                    if not in_category:
                        label = 'OTHER'
            tokens.append(label)
            tokens.append(node)
        relations[relation].append(tuple(tokens))

print('done analyzing conceptnet.')

with open('datasets/relations-with-categories.txt', 'w') as out_file:
    for relation, instances in relations.items():
        for tokens in instances:
            print('%s, %s, %s, %s, %s' % tokens, file=out_file)

print('done printing to file.')