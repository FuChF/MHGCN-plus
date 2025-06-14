# MHGCN-plus
The code of MHGCN+
## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0

## Datasets
### Link
The used datasets are available at:
* Alibaba https://tianchi.aliyun.com/competition/entrance/231719/information/
* Amazon http://jmcauley.ucsd.edu/data/amazon/
* Aminer https://github.com/librahu/
* IMDB https://github.com/seongjunyun/Graph_Transformer_Networks
* DBLP https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0

### Preprocess
We compress the data set into a mat format file, which includes the following contents.
* edges: array of subnetworks after coupling, each element in the array is a subnetwork.
* features: attributes of each node in the network.
* labels: label of labeled points.
* train: index of training set points for node classification. 
* valid: index of validation set points for node classification.
* test: index of test set points for node classification.

In addition, we also sample the positive and negative edges in the network, and divide them into three text files: train, valid and test for link prediction.

## Usage
First, you need to determine the dataset. If you want to do node classification tasks, you need to modify the data set path in `Node_classification.py`. If you want to do link prediction, you need to modify the dataset path in `Link_prediction.py`.

Second, you need to modify the number of weights in `Model.py`. The number of weights should be the number of sub-networks after decoupling.

Finally, you need to determine the sub-network and the number of sub-networks in `Decoupling_matrix_aggregation.py` and 'node_classfication_evaluate.py' or 'link_prediction_evaluate.py'.

Execute the following command to run the node classification task:

* `python python Node_Classification.py`

Execute the following command to run the link prediction task:

* `python Link_Prediction.py`

## Citation
If you find our paper and code useful in your research, please consider giving a star and citation :)

@article{10.1145/3650046,

author = {Fu, Chaofan and Yu, Pengyang and Yu, Yanwei and Huang, Chao and Zhao, Zhongying and Dong, Junyu},

title = {MHGCN+: Multiplex Heterogeneous Graph Convolutional Network},

year = {2024},

issue_date = {June 2024},

publisher = {Association for Computing Machinery},

address = {New York, NY, USA},

volume = {15},

number = {3},

journal = {ACM Trans. Intell. Syst. Technol.},

month = apr,

articleno = {51},

numpages = {25},

}
