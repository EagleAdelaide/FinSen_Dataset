import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Step 1: Download the dataset
train_iter, test_iter = AG_NEWS(split=('train', 'test'))

# Step 2: Tokenization
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Step 3: Build Vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Step 4: Text pipeline to convert text to tensors
def text_pipeline(x):
    return vocab(tokenizer(x))

# Label pipeline to convert labels to tensors
def label_pipeline(x):
    return int(x) - 1

# Step 5: Collate function to combine the text and labels into batches
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=0)
    return label_list, text_list, lengths

# Step 6: Prepare DataLoader
train_iter, test_iter = AG_NEWS(split=('train', 'test'))  # Reload iterators since we consumed them
train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=8, shuffle=False, collate_fn=collate_batch)

# Example usage: Iterate through the DataLoader
# for labels, texts, lengths in train_dataloader:
#     print("Labels:", labels)
#     print("Texts:", texts)
#     print("Lengths:", lengths)
#     break
