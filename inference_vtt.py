import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
import h5py
from typing import Tuple, Optional, Union
import numpy as np
sys.path.append('pycocoevalcap/')
sys.path.append('pycocoevalcap/bleu')
sys.path.append('pycocoevalcap/cider')
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from train_vtt import ClipCaptionModel, ClipCaptionPrefix, MSRVTTDataset, MappingType

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


def load_model(args, model_dir: str, epoch_or_latest: Union[str, int] = '_latest'):
    prefix_dim = 640 if args.is_rn else 512
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(model_dir, f"{args.prefix}{epoch_or_latest}.pt")
    model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, if_cross=args.cross)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)
    return generated_list[0]

def evaluate(dataset: MSRVTTDataset, model: ClipCaptionModel, use_beam_search):
    device = torch.device('cuda:0')
    batch_size = 1
    model = model.to(device)
    model.eval()
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    prediction_list = {}
    for tokens, mask, prefix, videoids in tqdm(val_dataloader):
        model.zero_grad()
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        with torch.no_grad():
            prefix_embed = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
        for idx, vid in enumerate(videoids):
            if use_beam_search:
                generated_text_prefix = generate_beam(model, dataset.tokenizer, embed=prefix_embed[idx].unsqueeze(0))[0]
            else:
                generated_text_prefix = generate2(model, dataset.tokenizer, embed=prefix_embed[idx].unsqueeze(0))
            prediction_list["video"+str(vid.item())] = [generated_text_prefix]
    return prediction_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./CV_Project_Dataset/MSRVTT_TRNVAL_CLIP_FEATURES.hdf5')
    parser.add_argument('--caption', default='./CV_Project_Dataset/train_val_videodatainfo.json')
    parser.add_argument('--model_dir', default='./msrvtt-checkpoints')
    parser.add_argument('--cross', dest='cross', action='store_true')
    parser.add_argument('--prefix', default='msrvtt_prefix', help='prefix for saved filenames')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--prefix_length_clip', type=int, default=28, help='set to number of frames')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_beam_search', dest='use_beam_search', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = MSRVTTDataset(args.data, args.caption, prefix_length=prefix_length, split="val")
    reference_list = json.load(open(f"./CV_Project_Dataset/val_dict.json"))
    # dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    model = load_model(args, args.model_dir, args.epoch-1)
    prediction_list = evaluate(dataset, model, use_beam_search=args.use_beam_search)
    save_result_name ="pred_beam.json" if args.use_beam_search else "pred.json"
    json.dump(prediction_list, open(os.path.join(args.model_dir, save_result_name), 'w'))
    score = get_score(reference_list, prediction_list)
    # print(score)
    json.dump(score,  open(os.path.join(args.model_dir, "score"+save_result_name), 'w'))



if __name__ == '__main__':
    main() 
           
       
