from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import tensorflow as tf
from tensorboard.plugins import projector

import re
import os
from tqdm import tqdm

model = GPT2LMHeadModel.from_pretrained('gpt2')

word_embeddings = model.transformer.wte.weight      # Word Token Embeddings
position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings

print(word_embeddings.shape)

print(position_embeddings.shape)

# create logging directory
log_dir='./logs/vocab/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

tokenizer.pretrained_vocab_files_map

dict(list(tokenizer.vocab.items())[:10])

vocab_list = sorted(tokenizer.vocab.items(), key=lambda x:x[1])

for k,v in tokenizer.vocab.items():
    if v < 10:
        print(k, v)

vocab_list[:10]



# Save the metadata file with "Ġ" replaced by "|"
with open(os.path.join(log_dir, 'metadata.tsv'), "w", encoding='utf-8') as f:
    for word, idx in tqdm(vocab_list):
        cleaned_word = word.replace("Ġ", "*")  # Replace Ġ with |
        f.write(f"{cleaned_word}\n")


embeddings = tf.Variable(model.transformer.wte.weight.detach().numpy())
checkpoint = tf.train.Checkpoint(embedding=embeddings)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()

# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

# %load_ext tensorboard
# %tensorboard --logdir ./logs/vocab/


