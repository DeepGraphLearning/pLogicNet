# pLogicNet
This is an implementation of the model from the paper [Probabilistic Logic Neural Networks for Reasoning](https://arxiv.org/abs/1906.08495).

## Usage
In our repo, four benchmark datasets are provided, including FB15k, FB15k-237, WN18, WN18RR. Those datasets are available in the ```data``` folder. The folder ```kge``` provides the codes for knowledge graph embedding, and the folder ```mln``` gives an implementation of the Markov logic network, in which four rule patterns are considered, including the composition rule, symmetric rule, inverse rule and subrelation rule.

Since the MLN module is written in C++, we need to compile the MLN codes before running the program. To compile the codes, we can enter the ```mln```  folder and execute the following command:
```
g++ -O3 mln.cpp -o mln -lpthread
```
Afterwards, we can run pLogicNet by using the script ```run.py``` in the main folder.

During training, the program will create a saving folder in ```record``` to save the intermediate outputs and the results, and the folder is named as the time when the job is submitted. For each iteration, the program will create a subfolder inside the saving folder. In each subfolder, the result of pLogicNet on validation set, the result of pLogicNet on test set and the result of pLogicNet* on test set are saved into ```result_kge_valid.txt```, ```result_kge.txt``` and ```result_kge_mln.txt``` respectively. Based on the validation results, we can then pick up the best model, and use it for evaluation or apply it to other knowledge graphs for link prediction.

## Acknowledgement
The knowledge graph embedding codes in the ```kge``` folder are from the nice repo [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding), where many knowledge graph embedding algorithms are implemented.

## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!
```
@inproceedings{qu2019probabilistic,
  title={Probabilistic Logic Neural Networks for Reasoning},
  author={Qu, Meng and Tang, Jian},
  booktitle={Advances in neural information processing systems},
  year={2019}
}
```


