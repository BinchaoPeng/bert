from allennlp.modules.elmo import Elmo, batch_to_ids
from time import time

# [batch_size, seq_len, embedding_dim=256]
options_file = "pre-model/elmo_model/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "pre-model/elmo_model/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

# [batch_size, seq_len, embedding_dim=1024]
options_file = "pre-model/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "pre-model/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, requires_grad=False, dropout=0.5)
time_since = time()
# use batch_to_ids to convert sentences to character ids
sentence_lists = ["AT"*2500, "AT GC"*1000, "ACN"*1000]
character_ids = batch_to_ids(sentence_lists)
print(character_ids.shape)
print(type(character_ids), "\n\n")

embeddings = elmo(character_ids)['elmo_representations']
print(embeddings)
print(type(embeddings))
print(type(embeddings[0]))
print(embeddings[0].shape)
time_end = time()

print((time_end - time_since))
"""
torch.Size([3, 3000, 50])
<class 'list'>
<class 'torch.Tensor'>
torch.Size([3, 3000, 1024])
"""
