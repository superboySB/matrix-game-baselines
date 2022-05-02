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
                 [-12, -2, -2],
                 [-12, -2, -2]]
```

#### Results

QMIX:
