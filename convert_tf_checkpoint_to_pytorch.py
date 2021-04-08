"""
convert_tf_checkpoint_to_pytorch.py 所在的路径 \Python\Lib\site-packages\pytorch_transformers
1. 在cmd里cd G:\Python\Lib\site-packages\pytorch_transformers，进入对应路径下；

2. 输入python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=E:/bert/chinese_L-12_H-768_A-12/bert_model.ckpt --bert_config_file=E:/bert/chinese_L-12_H-768_A-12/bert_config.json --pytorch_dump_path=E:/bert/chinese_L-12_H-768_A-12/pytorch_model.bin
*注：E:/bert/chinese_L-12_H-768_A-12 是bert预训练模型解压后的目录路径*

3. 运行一次这个转换脚本，可以得到一个PyTorch版本模型。可以忽略TensorFlow checkpoint(以bert_model.ckpt开头的三个文件)，但要保留配置文件(bert_config.json)和词汇表文件(vocab.txt)，因为PyTorch模型也需要这些文件。将bert_config.json复制粘贴重命名为config.json，否则执行pytorch_transformers代码会报错。

"""

# In[]:

# python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=E:/bert/chinese_L-12_H-768_A-12/bert_model.ckpt --bert_config_file=E:/bert/chinese_L-12_H-768_A-12/bert_config.json --pytorch_dump_path=E:/bert/chinese_L-12_H-768_A-12/pytorch_model.bin


# In[]:
