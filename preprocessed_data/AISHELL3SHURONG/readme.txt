
首先应该对原来已经跑的进行备份


2021年05月05日 星期三 15点32分
/home/aone/lisen/code/FastSpeech2/preprocessed_data/AISHELL3SHURONG/shurong_half/ 其中的train.txt是 /home/aone/lisen/code/FastSpeech2/preprocessed_data/AISHELL3SHURONG/train.txt中修改而来
修改部分为shurong部分的train部分减半, 具体实现方式为
从shurong_train.txt随机选择一半数据,然后添加到train_without_shurong.txt文件后，再复制过去

2021年05月09日 星期天 14点25分
