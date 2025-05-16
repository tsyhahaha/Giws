import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def batch_test_perplexity(model, dataloader, device, trg_pad_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)              # [B, L]
            trg = batch['trg'].to(device)              # [B, L]

            output = model(src=src, trg=trg[:, :-1])   # [B, L-1, vocab]
            target = trg[:, 1:]                        # [B, L-1]

            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                target.contiguous().view(-1),
                ignore_index=trg_pad_idx,
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += (target != trg_pad_idx).sum().item()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl.item()

def batch_test_bleu(model, dataloader, device, tokenizer, max_len=64):
    model.eval()
    references = []
    hypotheses = []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)        # [B, L]
            trg = batch['trg'].to(device)        # [B, L]

            # assuming inference functions "generate" exists
            generated = model.generate(src, max_len=max_len)  # [B, L]

            for ref_seq, hyp_seq in zip(trg, generated):
                ref_tokens = tokenizer.decode(ref_seq.tolist(), remove_special=True)
                hyp_tokens = tokenizer.decode(hyp_seq.tolist(), remove_special=True)
                
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)
    return bleu_score
