import json
import sys
import os
sys.path.append('pycocoevalcap/')
sys.path.append('pycocoevalcap/bleu')
sys.path.append('pycocoevalcap/cider')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor

def get_score(ref, pred):
        """
        ref, dictionary of reference sentences (id, sentence)
        pred, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        example_scores = {}
        for scorer, method in scorers:
            score, scores, vids = scorer.compute_score(ref, pred)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
                example_scores[method[-1]] = vids
            else:
                final_scores[method] = score
                example_scores[method] = vids
        return final_scores, example_scores


reference_list = json.load(open(f"./CV_Project_Dataset/val_dict.json", 'r'))
pred_dir = sys.argv[1]
prediction_list = json.load(open(pred_dir, 'r'))

scores, examples = get_score(reference_list, prediction_list)
print(scores)
shows = {}
# print(examples)
for m, exs in examples.items():
    shows[m] = [ [ex[0], reference_list[ex[0]][0], prediction_list[ex[0]][0], ex[1]] for ex in exs ]
dump_dir = os.path.join(os.path.dirname(pred_dir), "score+example"+os.path.basename(pred_dir))
json.dump([scores,shows], open(dump_dir, 'w'))
sys.stdout.flush()

print("test ended")
