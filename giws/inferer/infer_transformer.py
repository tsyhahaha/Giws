import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import logging
from giws.models import Transformer
from giws.data import TranslationDataset

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logger = logging.getLogger(__name__)

def compute_bleu(references, hypotheses):
    """
    references: List[List[str]]  # list of reference tokens
    hypotheses: List[str]        # predicted token list
    """
    smoothie = SmoothingFunction().method4
    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    pred = pred.view(-1, pred.size(-1))

    loss = F.cross_entropy(
                    pred,
                    gold,
                    ignore_index=trg_pad_idx,
                    reduction='sum',
                    label_smoothing=0.1 if smoothing else 0.0,
                )
    
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def patch_trg(trg):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def decode_tokens(token_ids, vocab, pad_idx):
    """ Converts a list of token ids to list of words, excluding padding and <eos> """
    tokens = []
    for idx in token_ids:
        if idx == pad_idx or idx == vocab.get('<eos>', None):
            break
        tokens.append(vocab[idx])  # vocab 是 Vocab 类对象
    return tokens


def setup_model(args):
    # single-GPU inference 
    device = torch.device(args.gpu_id) if args.use_gpu else 'cpu'
    save_info = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model = Transformer(
        **save_info['cfg'],
        max_length=args.max_len,
        device=device,
    )
    model.load_state_dict(save_info["model"])

    model.to(device)
    return model

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        items = [item[key] for item in batch if item[key] is not None]
        if items:
            processed_batch[key] = torch.stack(items)
        else:
            processed_batch[key] = None
    return processed_batch

def setup_dataset(args):
    test_dataset = TranslationDataset(
        chinese_file=os.path.join(args.data_path, 'cn.txt'),
        english_file=os.path.join(args.data_path, 'en.txt'),
        chinese_vocab_file=os.path.join(args.vocab_path, 'vocab.json'),
        english_vocab_file=os.path.join(args.vocab_path, 'vocab.json'),
        max_len=args.max_len,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info(f'Dataloader setup finish: test {len(test_dataset)}')
    return test_loader

@torch.no_grad()
def test(model, validation_data, device, pad_idx=[0, 0], trg_vocab=None):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    id2word = None
    if trg_vocab is not None:
        id2word = {v: k for k,v in trg_vocab.items()}
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    references = []
    hypotheses = []

    for batch in validation_data:
        # prepare data
        src_seq = batch['src'].to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch['trg']))

        # forward
        pred = model(src_seq, trg_seq)
        loss, n_correct, n_word = cal_performance(
            pred, gold, pad_idx[1], smoothing=False)

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        # === decode prediction and gold for BLEU ===
        pred_tokens = pred.argmax(-1)  # (batch, seq_len)
        for pred_seq, gold_seq in zip(pred_tokens, batch['trg']):
            pred_str = decode_tokens(pred_seq.tolist(), id2word, pad_idx[1])
            gold_str = decode_tokens(gold_seq[1:].tolist(), id2word, pad_idx[1])  # skip <bos>
            hypotheses.append(pred_str)
            references.append([gold_str])  # corpus_bleu expects list of list

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    ppl = torch.exp(torch.tensor(loss_per_word))

    bleu = compute_bleu(references, hypotheses)

    return loss_per_word, ppl.item(), accuracy, bleu



def inference(args):
    device = torch.device(args.gpu_id)

    # dataset setup
    test_dataloader = setup_dataset(args)
    word2idx = test_dataloader.dataset.get_word2idx(target='trg')

    model_path = args.model_path
    if os.path.isdir(model_path):
        model_paths = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.ckpt')]
    else:
        model_paths = [model_path]

    model_paths = sorted(model_paths, key=lambda x: int(x.split('.')[-2].split('_')[-1]))

    for model_path in model_paths:
        logger.info(f"Inference model: {model_path}")
        args.model_path = model_path
        model = setup_model(args)
        model.eval()

        loss, ppl, acc, bleu4 = test(model, test_dataloader, device, trg_vocab=word2idx)
        logger.info(f"Inference finished: "
                    f"loss = {loss} "
                    f"ppl = {ppl} "
                    f"acc = {acc} "
                    f"bleu4 = {bleu4*100:.2f}%")



    