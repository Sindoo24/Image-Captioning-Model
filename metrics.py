import evaluate
import nltk


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class MetricsCalculator:
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")

    def compute_metrics(self, predictions, references):
        bleu_score = self.bleu.compute(predictions=predictions, references=references)
        meteor_score = self.meteor.compute(predictions=predictions, references=references)
        rouge_score = self.rouge.compute(predictions=predictions, references=references)

        return {
            "BLEU": bleu_score['bleu'],
            "METEOR": meteor_score['meteor'],
            "ROUGE-L": rouge_score['rougeL']
        }