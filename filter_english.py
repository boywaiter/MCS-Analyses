# Filter out english relations out of conveptnet

lines = set([])
with open('datasets/conceptnet-assertions-5.7.0.csv', 'r') as inf:
    for line in inf:
        node_1, node_2 = line.split()[2:4]
        try:
            language_1 = node_1.split('/')[2]
            language_2 = node_2.split('/')[2]
            if language_1 == 'en' and language_2 == 'en':
                lines.add(line)
        except:
            continue

with open('datasets/simplified_english_conceptnet.csv', 'w') as outf:
    for line in lines:
        try:
            tokens = line.split()
            relation = tokens[1].split('/')[2]
            node_1 = tokens[2].split('/')[3]
            node_2 = tokens[3].split('/')[3]
            weight = tokens[-1][:-1]
            print(node_1, relation, node_2, weight, file=outf)
        except:
            continue
