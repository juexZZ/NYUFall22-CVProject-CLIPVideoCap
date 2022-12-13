import json
import sys

sys.path.append('pycocoevalcap/')
sys.path.append('pycocoevalcap/bleu')
sys.path.append('pycocoevalcap/cider')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def get_score(ref, pred):
    """
    ref, dictionary of reference sentences (id, sentence)
    pred, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
    """
    scorers = [
        # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, pred)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


reference_list = json.load(open(f"./CV_Project_Dataset/val_dict.json", 'r'))
prediction_list = json.load(open("./msrvtt-train/pred.json", 'r'))

scores = get_score(reference_list, prediction_list)
print(scores)
sys.stdout.flush()

print("test ended")
