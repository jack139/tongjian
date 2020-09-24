
## tongjian - a nlp test

### KNN 

训练
```shell
python3 train.py train_file model_name
```

测试
```shell
python3 predict.py "test_txt" model_name
```


### LSTM

训练
```shell
python3 -m lstm.gen_chn train model data/ts300.txt
```

生成文本
```shell
python3 -m lstm.gen_chn gen model 测试
```
