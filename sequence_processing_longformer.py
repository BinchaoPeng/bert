import numpy as np
from transformers import AutoTokenizer, AutoModel
# from transformers import pipeline  # !!! can not import pipline.......


def get_data(enhancers, promoters):
    model_name = 'pre-model/' + 'longformer-encdec-base-16384'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

    encoded_inputs = tokenizer(enhancers, return_tensors='pt', padding=True)
    X_en_feature = model(**encoded_inputs, return_netsors='pt')

    encoded_inputs = tokenizer(promoters, return_tensors='pt', padding=True)
    X_pr_feature = model(**encoded_inputs, return_netsors='pt')
    # print(type(feature))
    # feature = torch.as_tensor(feature)
    # print(feature)
    print(X_en_feature.shape)
    print(X_pr_feature.shape)
    X_en = np.array(X_en_feature)
    X_pr = np.array(X_pr_feature)
    print(X_en_tra)
    print(X_pr_tra)
    return X_en, X_pr


# In[]:


names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
name = names[0]
train_dir = 'data/%s/train/' % name
imbltrain = 'data/%s/imbltrain/' % name
test_dir = 'data/%s/test/' % name
Data_dir = 'data/%s/' % name
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


X_en_tra, X_pr_tra = get_data(enhancers_tra, promoters_tra)
# X_en_imtra, X_pr_imtra = get_data(im_enhancers_tra, im_promoters_tra)
# X_en_tes, X_pr_tes = get_data(enhancers_tes, promoters_tes)

# np.savez(Data_dir + '%s_train.npz' % name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
# np.savez(Data_dir + 'im_%s_train.npz' % name, X_en_tra=X_en_imtra, X_pr_tra=X_pr_imtra, y_tra=y_imtra)
# np.savez(Data_dir + '%s_test.npz' % name, X_en_tes=X_en_tes, X_pr_tes=X_pr_tes, y_tes=y_tes)
