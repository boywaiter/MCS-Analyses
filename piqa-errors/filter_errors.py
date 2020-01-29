with open("dev-labels.lst", "r") as labels:	
	l = labels.readlines()
	with open("dev-predictions.lst", "r") as pred:
		p = pred.readlines()
		with open("dev/dev.jsonl", "r") as data:
			d = data.readlines()
			with open('errors.txt', "w") as out_error:
				with open('corrects.txt', 'w') as out_correct:
					for i, label in enumerate(l):
						if label.strip() != p[i].strip():
							out_error.write(d[i])
						else:
							out_correct.write(d[i])
