import json
import pprint

dataset = []

with open("dev-labels.lst", "r") as labels:	
	with open("dev-predictions.lst", "r") as pred:
		with open("dev-probabilities.lst", "r") as prob:
			with open("dev/dev.jsonl", "r") as data:
				l = labels.readlines()
				p = pred.readlines()
				pr = prob.readlines()
				d = data.readlines()
				for i, label in enumerate(l):
					correct = label.strip() == p[i].strip()
					json_d = json.loads(d[i])
					# confidence as difference between probabilities for each prediction
					confidence = abs(float(pr[i].split()[0].strip()) - float(pr[i].split()[1].strip()))
					goal = json_d['goal']
					sol1 = json_d['sol1']
					sol2 = json_d['sol2']
					datapoint = {
						'goal': goal,
						'sol1': sol1,
						'sol2': sol2,
						'label': label.strip(),
						'prediction': p[i].strip(),
						'confidence': confidence
					}
					dataset.append(datapoint)

for datapoint in dataset:
	sol1_tokens = set(datapoint['sol1'].split())
	sol2_tokens = set(datapoint['sol2'].split())
	dif1 = sol1_tokens.difference(sol2_tokens)
	dif2 = sol2_tokens.difference(sol1_tokens)

	datapoint['dif1'] = dif1
	datapoint['dif2'] = dif2

incorrect_predictions = [t for t in dataset if t['label'] != t['prediction']]
correct_predictions = [t for t in dataset if t['label'] == t['prediction']]

# On average, there are 2.5 words different between two answers
print(sum([len(t['dif1']) for t in dataset])/len(dataset))
print(sum([len(t['dif2']) for t in dataset])/len(dataset))
print(sum([len(t['dif1']) for t in correct_predictions])/len(correct_predictions))
print(sum([len(t['dif2']) for t in correct_predictions])/len(correct_predictions))
print(sum([len(t['dif1']) for t in incorrect_predictions])/len(incorrect_predictions))
print(sum([len(t['dif2']) for t in incorrect_predictions])/len(incorrect_predictions))


# # sort based on confidence
correct_predictions.sort(key = lambda token: token['confidence'])
incorrect_predictions.sort(key = lambda token: token['confidence'])

# printer = pprint.PrettyPrinter()
# for t in incorrect_predictions:
# 	printer.pprint(t)