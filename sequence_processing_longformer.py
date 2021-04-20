import numpy as np
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig


# from transformers import pipeline  # !!! can not import pipline.......


def get_data(enhancers, promoters):
    model_name = 'pre-model/' + 'longformer-base-4096'
    config = LongformerConfig.from_pretrained(model_name)
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name, config=config, tokenizer=tokenizer)

    # X_enpr_features = []
    # for enhancer, promoter in zip(enhancers, promoters):
    #     encoded_inputs = tokenizer(enhancer + tokenizer.sep_token + promoter, return_tensors='pt', padding=True)
    #     X_enpr_feature = model(**encoded_inputs, return_netsors='pt')
    #     X_enpr_features.append(X_enpr_feature)

    encoded_inputs = tokenizer(enhancers + tokenizer.sep_token + promoters, return_tensors='pt', padding=True)
    X_enpr_features = model(**encoded_inputs, return_netsors='pt')
    X_enpr = np.array(X_enpr_features)
    return X_enpr


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

X_enpr_tra = get_data(enhancers_tra, promoters_tra)
X_enpr_imtra = get_data(im_enhancers_tra, im_promoters_tra)
X_enpr_tes = get_data(enhancers_tes, promoters_tes)

np.savez(Data_dir + '%s_train.npz' % name, X_enpr_tra=X_enpr_tra, y_tra=y_tra)
np.savez(Data_dir + 'im_%s_train.npz' % name, X_enpr_tra=X_enpr_imtra, y_tra=y_imtra)
np.savez(Data_dir + '%s_test.npz' % name, X_enpr_tes=X_enpr_tes, y_tes=y_tes)
