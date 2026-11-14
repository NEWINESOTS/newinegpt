"""LoRA fine-tune on a rented GPU (transformers + peft), loss on the final answer only; then merge, test generations.
Same data and prompt format as the Mac/MLX config (train/data, train/prompt.py).
python pod_train.py --base Qwen/Qwen3-4B --data /root/data --out /root/out/qwen3-4b-v1 --merged /root/merged [--epochs 2]"""
import argparse, json, math, os, random, time
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--merged', required=True)
ap.add_argument('--epochs', type=float, default=2); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--alpha', type=int, default=32); ap.add_argument('--bs', type=int, default=8); ap.add_argument('--accum', type=int, default=2)
ap.add_argument('--maxlen', type=int, default=2048); ap.add_argument('--n_test', type=int, default=200)
ap.add_argument('--adapter_only', action='store_true', help='stop after saving the adapter (merge + tests + GGUF are done on the Mac)')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.manual_seed(2026); random.seed(2026)
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def encode(msgs):
    """Full conversation, labels only on the final assistant turn (its '<think>\\n\\n</think>\\n\\n' prefix is part of the prompt)."""
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    assert full.startswith(prompt), 'chat template: prompt is not a prefix of the full conversation'
    p = tok(prompt, add_special_tokens=False)['input_ids']; f = tok(full, add_special_tokens=False)['input_ids']
    assert f[:len(p)] == p
    f = f[:a.maxlen]; lab = [-100] * min(len(p), len(f)) + f[len(p):]
    return f, lab

class DS(Dataset):
    def __init__(self, path):
        self.rows = [encode(json.loads(l)['messages']) for l in open(path)]
        self.rows = [r for r in self.rows if any(x != -100 for x in r[1])]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return {'input_ids': self.rows[i][0], 'labels': self.rows[i][1]}
def collate(b):
    b = [(x['input_ids'], x['labels']) for x in b]
    n = max(len(x[0]) for x in b)
    ids = torch.full((len(b), n), tok.pad_token_id); lab = torch.full((len(b), n), -100); att = torch.zeros((len(b), n), dtype=torch.long)
    for i, (x, y) in enumerate(b): ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return {'input_ids': ids, 'labels': lab, 'attention_mask': att}

t0 = time.time()
tr, va = DS(f'{a.data}/train.jsonl'), DS(f'{a.data}/valid.jsonl')
print(f'encoded train {len(tr)} valid {len(va)} ({time.time()-t0:.0f}s); answer-token share {sum(sum(1 for y in r[1] if y != -100) for r in tr.rows) / sum(len(r[0]) for r in tr.rows):.2%}', flush=True)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, task_type='CAUSAL_LM',
                                         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
model.print_trainable_parameters()
args = TrainingArguments(output_dir=a.out + '/ckpt', num_train_epochs=a.epochs, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
                         per_device_eval_batch_size=a.bs, learning_rate=a.lr, lr_scheduler_type='cosine', warmup_ratio=0.03, weight_decay=0.0,
                         bf16=True, logging_steps=10, eval_strategy='steps', eval_steps=100, save_strategy='steps', save_steps=200, save_total_limit=2,
                         report_to=[], seed=2026, dataloader_num_workers=2, group_by_length=True, remove_unused_columns=False)
tr_ = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, data_collator=collate)
tr_.train()
ev = tr_.evaluate(); print('final eval', ev, flush=True)
model.save_pretrained(a.out + '/adapter'); tok.save_pretrained(a.out + '/adapter')
json.dump({'log_history': tr_.state.log_history, 'final_eval': ev, 'args': vars(a), 'train_rows': len(tr), 'valid_rows': len(va),
           'train_seconds': time.time() - t0}, open(a.out + '/train_log.json', 'w'), indent=1)

if a.adapter_only:
    print('adapter saved; skipping merge and test generations', flush=True); raise SystemExit(0)
# merge into the bf16 base for export
model = model.merge_and_unload(); model.save_pretrained(a.merged, safe_serialization=True); tok.save_pretrained(a.merged)
print('merged', a.merged, flush=True)

# held-out test generations (greedy), for the eval report
model.eval(); tok.padding_side = 'left'
test = [json.loads(l) for l in open(f'{a.data}/test.jsonl')]
random.Random(7).shuffle(test); test = test[:a.n_test]
with open(a.out + '/test_generations.jsonl', 'w') as f:
    for i in range(0, len(test), 16):
        part = test[i:i + 16]
        prompts = [tok.apply_chat_template(e['messages'][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in part]
        enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): out = model.generate(**enc, max_new_tokens=600, do_sample=False, pad_token_id=tok.pad_token_id)
        for e, o in zip(part, out):
            f.write(json.dumps({'type': e['type'], 'question': e['messages'][1]['content'], 'reference': e['messages'][2]['content'],
                                'output': tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()}, ensure_ascii=False) + '\n')
print('test generations written', flush=True)
