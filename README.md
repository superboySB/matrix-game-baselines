# matrix-game-baselines

###  1. Install

Install requirements

```shell
conda create -n matrix python==3.8
conda activate matrix
conda install pytorch torchvision torchaudio cudatoolkit=11.3 tensorboard future
pip install -r requirements.txt
```

### 2. Results in Matrix Games 

Train Algorithms (config names): 

- VDN（vdn, weighted_vdn）
- QMIX（qmix)  
- WQMIX（ow_qmix, cw_qmix）
- Qtran（vdn_qtran, qmix_qtran）
- Qplex（qplex）
- [MAIC](https://github.com/mansicer/MAIC)（qmix_maic, qplex_maic）

Train:

``` sh
# Step1: choose alg
# Step2: choose matrix
# Step3: choose other parameters
python q_family.py
```
#### Matrix

```
payoff_values = [[8, -12, -12],
                 [-12, 0, 0],
                 [-12, 0, 0]]
```

#### Results

VDN
```
predicted_values = [[-20.21, -10.12, -10.12],
                 [-10.12, 0.00, 0.00],
                 [-10.12, 0.00, 0.00]]
```

QMIX
```
predicted_values = [[-11.92, -11.92, -11.92],
                 [-11.92, 0.00, 0.00],
                 [-11.92, 0.00, 0.00]]
```

WQMIX（cw_qmix、ow_qmix）
```
predicted_values = [[8.00, -11.73, -11.73],
                 [-11.73, -11.73, -11.73],
                 [-11.73, -11.73, -11.73]]
```
```
predicted_values = [[8.00, -10.62, -10.62],
                 [-10.62, -10.62, -10.62],
                 [-10.62, -10.62, -10.62]]
```

Qtran
```
predicted_values = [[8.00, 3.99, 4.06],
                 [4.00, 0.00, 0.06],
                 [4.00, 0.00, 0.06]]
```

Qplex
```
predicted_values = [[3.94, -11.68, -11.74],
                 [-11.68, 0.21, 0.14],
                 [-11.75, 0.13, 3.97]]
```



## Reference 

* 【全网最佳入门知乎】请看田哥的拨云见日、剥丝抽茧式教学：https://zhuanlan.zhihu.com/p/421909836
* https://github.com/tjuHaoXiaotian/Qfamily_for_MatrixGame
