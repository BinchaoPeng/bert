from allennlp.modules.elmo import Elmo, batch_to_ids

# [3, 3000, 256]
# options_file = "pre-model/elmo_model/elmo_2x1024_128_2048cnn_1xhighway_options.json"
# weight_file = "pre-model/elmo_model/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

# [3, 3000, 1024]
options_file = "pre-model/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "pre-model/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 3, requires_grad=False, dropout=0.5)

# use batch_to_ids to convert sentences to character ids
sentence_lists = ["ATGCATGC", "ATGC", "ACNTGAGTCA" * 300]
character_ids = batch_to_ids(sentence_lists)
print(character_ids.shape)

embeddings = elmo(character_ids)['elmo_representations']
print(type(embeddings))
print(type(embeddings[0]))
print(embeddings[0].shape)
