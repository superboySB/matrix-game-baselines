# matrix-game-baselines

###  1. Install

Install requirements

```shell
conda create -n om python==3.8
conda activate om
conda install pytorch torchvision torchaudio cudatoolkit=11.3 tensorboard future
pip install -r requirements.txt
```

### 2. Results in Matrix Games 

Train Algorithms (config names): 
- Our solution（???）
- VDN（vdn）
- QMIX（qmix)  
- WQMIX（cw_qmix，ow_qmix）
- Qatten（qatten）
- Qtran（qtran）**x**
- Qplex（qplex）**x**
- MAIC（maic ，maic_qplex **x**）forked from https://github.com/mansicer/MAIC

Train:

``` sh
python3 main.py --config=maic --env-config=one_step_matrix_game with env_args.map_name=one_step_matrix_game
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
Qatten
```
predicted_values = [[-22.81, -11.44, -11.44],
                 [-11.44, 0.00, 0.00],
                 [-11.44, 0.00, 0.00]]
```

Qtran（refer to https://zhuanlan.zhihu.com/p/421909836）
```
predicted_values = [[8.00, 3.99, 4.06],
                 [4.00, 0.00, 0.06],
                 [4.00, 0.00, 0.06]]
```

Qplex（refer to https://zhuanlan.zhihu.com/p/421909836）
```
predicted_values = [[3.94, -11.68, -11.74],
                 [-11.68, 0.21, 0.14],
                 [-11.75, 0.13, 3.97]]
```

MAIC（maic，maic_qplex ）
```
predicted_values = [[-11.13, -11.13, -11.13],
                 [-11.13, 0, 0],
                 [-11.13, 0, 0]]
```
```
predicted_values = TODO!
```



## Reference 

【全网最佳入门知乎】请看田哥的拨云见日、剥丝抽茧式教学：https://zhuanlan.zhihu.com/p/421909836