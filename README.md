# Counterfactual Graph Learning for Anomaly Detection on Attributed Networks

This is a repository hosting the code of our paper: [Counterfactual Graph Learning for Anomaly Detection on Attributed Networks](https://github.com/ChunjingXiao/CFAD/blob/main/TKDE_23_CFAD.pdf), IEEE Transactions on Knowledge and Data Engineering, 2023, 35(10):10540 - 10553. https://ieeexplore.ieee.org/abstract/document/10056298
 

# Citation

@article{xiao2023counterfactual,  
&nbsp; &nbsp;author={Xiao, Chunjing and Xu, Xovee and Lei, Yue and Zhang, Kunpeng and Liu, Siyuan and Zhou, Fan},  
&nbsp; &nbsp;journal={IEEE Transactions on Knowledge and Data Engineering},   
&nbsp; &nbsp;title={Counterfactual Graph Learning for Anomaly Detection on Attributed Networks},   
&nbsp; &nbsp;year={2023},  
&nbsp; &nbsp;volume={35},  
&nbsp; &nbsp;number={10},  
&nbsp; &nbsp;pages={10540-10553},  
}

## Data

- The data is in directory graphs.    

## Dependencies

Run the following command to install dependencies with Anaconda virtual environment:
```shell
conda create -n cfad python==3.9

pip install -r requirements.txt
```

## Run

```shell
# PubMed
python run.py

```

Description of hyper-parameters can be found in `run.py`.
