import torch
import json
from collections import defaultdict


class GlossTokenizer:
    def __init__(self, tokenizer_cfg):
        with open(tokenizer_cfg["gloss2id_file"], "r") as f:
            self.gloss2id = json.load(f)
        self.gloss2id = defaultdict(lambda: self.gloss2id["<unk>"], self.gloss2id)

        self.id2gloss = {v: k for k, v in self.gloss2id.items()}
        self.id2gloss = defaultdict(lambda: self.id2gloss["<unk>"], self.id2gloss)

        self.lower_case = False
        self.split = tokenizer_cfg.get("split", " ")

        if "<s>" in self.gloss2id:
            self.start_token = "<s>"
            self.start_id = self.gloss2id[self.start_token]

        if "<pad>" in self.gloss2id:
            self.pad_token = "<pad>"
            self.pad_id = self.gloss2id[self.pad_token]
        else:
            raise ValueError("pad token not in gloss2id")

        self.special_tokens = [self.start_token, self.pad_token, "<unk>"]

    def encode(self, _input, max_len=None, has_split=True, return_length=False):
        if not has_split:
            _input = _input.replace("  ", " ")
            _input = _input.split(self.split)
        attention_mask = torch.ones(len(_input), dtype=torch.long)
        inputs_ids = torch.tensor(
            [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in _input]
        )
        if max_len is not None:
            attention_mask = torch.concat(
                (attention_mask, torch.zeros(max_len - len(_input), dtype=torch.long)),
                dim=0,
            )

            inputs_ids = torch.concat(
                (
                    inputs_ids,
                    torch.ones(max_len - len(_input), dtype=torch.long) * self.pad_id,
                ),
                dim=0,
            )

        return {
            "input_ids": inputs_ids,
            "attention_mask": attention_mask,
            "length": len(_input),
        }

    def batch_encode(self, batch, max_len=None, return_length=False):
        batch = [x.split(self.split) for x in batch]
        if max_len is None:
            max_len = max([len(x) for x in batch])

        input_ids, attention_mask = [], []

        if return_length:
            lengths = []

        for seq in batch:
            output = self.encode(seq, max_len, return_length=return_length)
            input_ids.append(output["input_ids"])
            attention_mask.append(output["attention_mask"])

            if return_length:
                lengths.append(output["length"])

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        if return_length:
            lengths = torch.tensor(lengths)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "length": lengths if return_length else None,
        }

    def decode(self, _input, skip_special_tokens=True):
        if type(_input) is dict:
            tokens = _input["input_ids"]

        elif "input_ids" in _input:
            tokens = _input["input_ids"]
        else:
            tokens = torch.tensor(_input)

        if skip_special_tokens:
            # Convert special_tokens list to tensor of token IDs for proper isin() usage
            special_tokens_ids = torch.tensor(
                [self.gloss2id[token] for token in self.special_tokens],
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = tokens[~torch.isin(tokens, special_tokens_ids)]
        tokens = tokens.tolist()
        return " ".join([self.id2gloss[int(x)] for x in tokens])

    def batch_decode(self, batch, skip_special_tokens=True):
        if type(batch) is dict:
            batch = batch["input_ids"]
        return [self.decode(x, skip_special_tokens) for x in batch]

    def __len__(self):
        return len(self.id2gloss)


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    cgf = {"gloss2id_file": "./data/Phoenix-2014/gloss2id.json", "split": " "}
    tokenizer = GlossTokenizer(cgf)

    df = pd.read_csv("./data/Phoenix-2014/annotations/all.csv", sep="|")

    sentences = df["annotation"].tolist()
    print(sentences[0])
    for s in tqdm(sentences):
        # Normalize input string by replacing multiple spaces with single space
        normalized_s = " ".join(s.split())
        output = tokenizer.encode(s, has_split=False)
        decode = tokenizer.decode(output, skip_special_tokens=True)
        # Normalize decoded string for comparison
        normalized_decode = " ".join(decode.split())
        assert normalized_s == normalized_decode, f"{s} != {decode}"
