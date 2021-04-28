import numpy as np
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig

# from transformers import pipeline  # !!! can not import pipline.......
print("test!")


def get_max_length(seq_list, tokenizer):
    max_value = 0
    for txt in seq_list:
        tokens = tokenizer.tokenize(txt)
        token_value = len(tokens)
        # print(tokens)
        # print(token_value)

        if token_value > max_value:
            max_value = token_value
        print("max_length is:", max_value)
        return max_value


def get_data_max_length(enhancers, promoters):
    ep_list = [enhancer + tokenizer.sep_token + promoter for enhancer, promoter in zip(enhancers, promoters)]
    max_length = get_max_length(ep_list, tokenizer)
    print("MAX:", max_length, "##" * 50)  # 2626 2659
    return max_length


def get_data(enhancers, promoters, tokenizer, model, max_length):
    X_enprs = []
    for enhancer, promoter in zip(enhancers, promoters):
        encoded_inputs = tokenizer(enhancer + tokenizer.sep_token + promoter, return_tensors='pt', padding=True,
                                   max_length=max_length)

        X_enpr_tensor = model(**encoded_inputs)[0]
        # print(X_enpr_features.shape)

        X_enpr_np = X_enpr_tensor.detach().numpy()
        X_enprs.append(X_enpr_np)

    # ep_list = [enhancer + tokenizer.sep_token + promoter for enhancer, promoter in zip(enhancers, promoters)]
    # encoded_inputs = tokenizer(ep_list, return_tensors='pt', padding=True)
    # X_enpr_features = model(**encoded_inputs, return_netsors='pt')
    print("add:", len(X_enprs))
    return X_enprs


def save_data(enhancers, promoters, tokenizer, model, max_length, file):
    with open(file, 'ab') as f:
        for enhancer, promoter in zip(enhancers, promoters):
            print(tokenizer.sep_token)
            encoded_inputs = tokenizer(enhancer + tokenizer.sep_token + promoter, return_tensors='pt', padding=True,
                                       max_length=max_length)

            X_enpr_tensor = model(**encoded_inputs)[0]
            # print(X_enpr_features.shape)
            X_enpr_np = X_enpr_tensor.detach().numpy()

            for item in X_enpr_np:
                # print(item)
                np.savetxt(f, item, delimiter=',')

    f.close()
    # ep_list = [enhancer + tokenizer.sep_token + promoter for enhancer, promoter in zip(enhancers, promoters)]
    # encoded_inputs = tokenizer(ep_list, return_tensors='pt', padding=True)
    # X_enpr_features = model(**encoded_inputs, return_netsors='pt')

    print("save over!")


# In[]:
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
name = names[0]
feature_name = "longformer-hug"
train_dir = 'data/%s/train/' % name
imbltrain = 'data/%s/imbltrain/' % name
test_dir = 'data/%s/test/' % name
Data_dir = 'data/%s/%s/' % (name, feature_name)
print('Experiment on %s dataset' % name)

print('Loading seq data...')
enhancers_tra = open(train_dir + '%s_enhancer.fasta' % name, 'r').read().splitlines()[1::2]
promoters_tra = open(train_dir + '%s_promoter.fasta' % name, 'r').read().splitlines()[1::2]
y_tra = np.loadtxt(train_dir + '%s_label.txt' % name)

im_enhancers_tra = open(imbltrain + '%s_enhancer.fasta' % name, 'r').read().splitlines()[1::2]
im_promoters_tra = open(imbltrain + '%s_promoter.fasta' % name, 'r').read().splitlines()[1::2]
y_imtra = np.loadtxt(imbltrain + '%s_label.txt' % name)

enhancers_tes = open(test_dir + '%s_enhancer_test.fasta' % name, 'r').read().splitlines()[1::2]
promoters_tes = open(test_dir + '%s_promoter_test.fasta' % name, 'r').read().splitlines()[1::2]
y_tes = np.loadtxt(test_dir + '%s_label_test.txt' % name)

print('平衡训练集')
print('pos_samples:' + str(int(sum(y_tra))))
print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))
print('不平衡训练集')
print('pos_samples:' + str(int(sum(y_imtra))))
print('neg_samples:' + str(len(y_imtra) - int(sum(y_imtra))))
print('测试集')
print('pos_samples:' + str(int(sum(y_tes))))
print('neg_samples:' + str(len(y_tes) - int(sum(y_tes))))

# In[ ]:
model_name = 'pre-model/' + 'longformer-base-4096'
config = LongformerConfig.from_pretrained(model_name)
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name, config=config)

a1 = get_data_max_length(enhancers_tra, promoters_tra)
a2 = get_data_max_length(im_enhancers_tra, im_promoters_tra)
a3 = get_data_max_length(enhancers_tes, promoters_tes)

max_value = max(a1, a2, a3)
print(max_value)
# X_enpr_tra = get_data(enhancers_tra, promoters_tra, tokenizer, model)
# np.savez(Data_dir + '%s_train.npz' % name, X_enpr_tra=X_enpr_tra, y_tra=y_tra)
# print("saved train!")
# save_data(enhancers_tra, promoters_tra, tokenizer, model, max_value, Data_dir + '%s_train.csv' % name)

#
# X_enpr_imtra = get_data(im_enhancers_tra, im_promoters_tra, tokenizer, model)
# np.savez(Data_dir + 'im_%s_train.npz' % name, X_enpr_tra=X_enpr_imtra, y_tra=y_imtra)
# print("saved im_train!")
# save_data(im_enhancers_tra, im_promoters_tra, tokenizer, model, max_value, Data_dir + 'im_%s_train.csv' % name)

# 
# X_enpr_tes = get_data(enhancers_tes, promoters_tes, tokenizer, model)
# np.savez(Data_dir + '%s_test.npz' % name, X_enpr_tes=X_enpr_tes, y_tes=y_tes)
# print("saved test!")
# save_data(enhancers_tes, promoters_tes, tokenizer, model, max_value, Data_dir + '%s_test.csv' % name)
