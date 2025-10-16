import evaluate

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(axis = -1)
    acc = accuracy.compute(predictions = predictions, references = labels)
    f1_score = f1.compute(predictions = predictions, references = labels, average = 'weighted')
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}