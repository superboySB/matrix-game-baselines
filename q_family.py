# -*- coding:UTF-8 -*-
# @Time    : 2022/5/08 21:00
# @Author  : HaoXiaotian, DaiZipeng
# @Contact : xiaotianhao@tju.edu.cn
# @File    : q_family.py
# TODO: 不考虑target network，没有double dueling Q

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math


class QFamily(nn.Module):
    def __init__(self, algo, agent_num, action_num, hidden_num):
        super(QFamily, self).__init__()
        self.algo = algo

        self.agent_num = agent_num
        self.action_num = action_num
        self.hidden_num = hidden_num
        self.latent_num = 8  # for MAIC

        # (agent_num, action_num)
        self.local_individual_qs = Parameter(torch.randn(1, agent_num, action_num))

        # matrix game没有state，所以不需要hyper net，似乎没有实现Qatten的必要
        if algo in ["vdn", "vdn_qtran"]:
            pass
        elif algo in "weighted_vdn":
            self.w1 = Parameter(torch.randn([agent_num, 1], dtype=torch.float32))
            self.b1 = Parameter(torch.randn([1, 1], dtype=torch.float32))
        elif algo in ["qmix", "qmix_qtran", "ow_qmix", "cw_qmix", "qmix_maic"]:
            # (agent_num, embedding_dim)
            self.w1 = Parameter(torch.randn([agent_num, hidden_num], dtype=torch.float32))
            self.b1 = Parameter(torch.randn([1, hidden_num], dtype=torch.float32))
            # (embedding_dim, 1)
            self.w2 = Parameter(torch.randn([hidden_num, 1], dtype=torch.float32))
            self.b2 = Parameter(torch.randn([1, 1], dtype=torch.float32))
            if algo in ["ow_qmix", "cw_qmix"]:
                self.alpha = 0.1
                self.central_loss_weight = 1
                # central feed-forward net, note that state is none in matrix games
                self.central_q_net = nn.Sequential(
                    nn.Linear(agent_num, hidden_num),
                    nn.ReLU(),
                    nn.Linear(hidden_num, 1),
                )
            if algo == "qmix_maic":
                self.msg_net = Parameter(torch.randn(1, agent_num, agent_num, action_num))  # 对角线不更新

        elif algo in ["qplex", "qplex_maic"]:
            # lambda net, note that state is none in matrix games
            self.lambda_net = nn.Sequential(
                nn.Linear(agent_num * action_num, hidden_num),
                nn.ReLU(),
                nn.Linear(hidden_num, agent_num),
            )
            if algo == "qplex_maic":
                self.msg_net = Parameter(torch.randn(1, agent_num, agent_num, action_num))
        else:
            raise NotImplementedError

        # init parameters
        self._init_parameters()

    def _init_parameters(self):
        # init individual q-value
        init.kaiming_uniform_(self.local_individual_qs, a=math.sqrt(5))
        # init.uniform_(self.individual_qs, 0, 0)
        print("******************* [q_i] init q tables *******************")
        q_print = self.local_individual_qs[0]  # [agent_num, action_num]
        for agent_idx in range(self.agent_num):
            individual_q = q_print[agent_idx]  # [action_num]
            print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                    individual_q.max(
                                                                                        dim=0)[1].item()))
            print(individual_q.tolist())
            print("--------------------------------------\n")

    def forward(self, batch_size, batch_action, q_joint, print_log=False):
        assert batch_action.shape[0] == batch_size
        assert batch_action.shape[1] == self.agent_num
        assert batch_action.shape[2] == 1

        # Get messages from team modeling
        if self.algo in ["qmix_maic", "qplex_maic"]:
            global_individual_qs = torch.zeros_like(self.local_individual_qs)
            for j in range(self.agent_num):  # 接收方
                for i in range(self.agent_num):  # 发送方
                    if i == j:
                        global_individual_qs[:, j, :] += self.local_individual_qs[:, j, :]
                    else:
                        global_individual_qs[:, j, :] += self.msg_net[:, i, j, :]
        else:
            global_individual_qs = self.local_individual_qs

        # Get action values (batch_size, agent_num)
        selected_individual_qs = torch.gather(
            global_individual_qs.expand(batch_size, self.agent_num, self.action_num),
            dim=2, index=batch_action.long()).squeeze(2)  # Remove the last dim

        # Calculate q_total
        if self.algo in ["vdn", "vdn_qtran"]:
            q_tot = selected_individual_qs.sum(dim=1, keepdim=True)
        elif self.algo == "weighted_vdn":
            q_tot = torch.mm(selected_individual_qs, torch.abs(self.w1)) + self.b1
        elif self.algo in ["qmix", "qmix_qtran", "ow_qmix", "cw_qmix", "qmix_maic"]:
            hidden = F.elu(torch.mm(selected_individual_qs, torch.abs(self.w1)) + self.b1)
            q_tot = torch.mm(hidden, torch.abs(self.w2)) + self.b2
            if self.algo in ["ow_qmix", "cw_qmix"]:
                q_central = self.central_q_net(selected_individual_qs)
                w_s = torch.ones_like(q_tot) * self.alpha
                if self.algo == "ow_qmix":
                    w_s = torch.where(q_tot < q_joint, torch.ones_like(q_tot), w_s)
                else:
                    q_upper, _ = global_individual_qs.max(dim=2, keepdim=False)
                    q_central_upper = self.central_q_net(q_upper)
                    # greedy actions
                    individual_greedy_action = global_individual_qs.max(dim=2, keepdim=True)[1]
                    # the sample in current batch in which the actions == greedy actions
                    max_point_mask = (individual_greedy_action == batch_action).long().sum(axis=1) == self.agent_num
                    w_s = torch.where((q_joint > q_central_upper) | max_point_mask, torch.ones_like(q_tot), w_s)

        elif self.algo in ["qplex", "qplex_maic"]:
            q_upper, _ = global_individual_qs.max(dim=2, keepdim=False)
            # (batch_size, agent_num)
            adv = selected_individual_qs - q_upper

            onehot_batch_action = torch.zeros((*batch_action.shape[:-1], self.action_num), dtype=torch.float32,
                                              device=batch_action.device). \
                scatter_(-1, batch_action, 1.0).view(batch_size, -1)  # [batch, agent_num * action_num]
            w_lambda = torch.abs(self.lambda_net(onehot_batch_action))  # 约束lambda>0

            v_tot = q_upper.sum(dim=1, keepdim=True)  # current maximum point
            adv_tot = (adv * w_lambda).sum(dim=1, keepdim=True)  # weighted sum
            q_tot = v_tot + adv_tot
        else:
            raise NotImplementedError

        if print_log:
            print("******************* [q_i] Learned individual q tables *******************")
            q_print = global_individual_qs[0]  # [agent_num, action_num]
            for agent_idx in range(self.agent_num):
                individual_q = q_print[agent_idx]  # [action_num]
                print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                        individual_q.max(
                                                                                            dim=0)[1].item()))
                print(individual_q.tolist())
                print("--------------------------------------\n")

        # get the greedy actions from each agents
        q_print = global_individual_qs[0]
        best_action_index=0
        for agent_idx in range(self.agent_num):
            individual_q = q_print[agent_idx]
            best_action_i=individual_q.max(dim=0)[1].item()
            best_action_index+=best_action_i*pow(action_num,agent_num-agent_idx-1)

        # Calculate loss
        if self.algo in ["vdn_qtran", "qmix_qtran"]:
            # greedy actions
            individual_greedy_action = global_individual_qs.max(dim=2, keepdim=True)[1]
            # the sample in current batch in which the actions == greedy actions
            max_point_mask = ((individual_greedy_action == batch_action).long().sum(axis=1) == self.agent_num).float()
            q_clip = torch.max(q_tot, q_joint).detach()  # Qtran核心： to ensure q_tot >= q_label
            loss = torch.mean(
                max_point_mask * ((q_tot - q_joint) ** 2) + (1 - max_point_mask) * ((q_tot - q_clip) ** 2))
        if self.algo in ["ow_qmix", "cw_qmix"]:
            loss = torch.mean(w_s * ((q_tot - q_joint) ** 2)) + self.central_loss_weight * torch.mean(
                (q_central - q_joint) ** 2)
        else:
            loss = torch.mean((q_tot - q_joint) ** 2)
        return q_tot, loss, best_action_index


def train():
    assert pow(action_num, agent_num) == len(payoff_flatten_vector)
    batch_size = len(payoff_flatten_vector)
    q_joint = torch.from_numpy(np.reshape(
        np.array(payoff_flatten_vector, dtype=np.float32)
        , newshape=[batch_size, 1])
    )
    action_index = np.arange(0, action_num)
    mesh_result = np.array(np.meshgrid(*[action_index for _ in range(agent_num)])).T.reshape(-1, agent_num)  # 笛卡儿积
    batch_action = torch.from_numpy(mesh_result.reshape(batch_size, agent_num, 1))  # 遍历action space，屏蔽探索影响

    one_step_q_network = QFamily(algo=algo, agent_num=agent_num, action_num=action_num, hidden_num=hidden_num)
    optimizer = torch.optim.Adam(params=one_step_q_network.parameters(), lr=0.01)

    for epoch in range(round):
        q_tot, loss, best_action_index = one_step_q_network.forward(batch_size, batch_action, q_joint)
        if epoch % 10 == 0:
            print(f"Iter={epoch}: MSE loss={loss.item()}, Optimal reward={q_joint[best_action_index].item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    q_tot, _ , _ = one_step_q_network.forward(batch_size, batch_action, q_joint, print_log=True)

    print("******************* Predicted Q_tot: *******************")
    q_print = q_tot.detach().tolist()
    for row in range(action_num):
        start_pos = row * pow(action_num, agent_num - 1)
        print(q_print[start_pos: start_pos + pow(action_num, agent_num - 1)])

    print("******************* True Q_joint: *******************")
    q_print = q_joint.detach().tolist()
    for row in range(action_num):
        start_pos = row * pow(action_num, agent_num - 1)
        print(q_print[start_pos: start_pos + pow(action_num, agent_num - 1)])


if __name__ == "__main__":
    ### TODO Step1: choose alg
    # algo = "vdn"
    # algo = "weighted_vdn"
    # algo = "qmix"
    # algo = "ow_qmix"
    # algo = "cw_qmix"   # consume more time than ow_qmix
    # algo = "vdn_qtran"
    # algo = "qmix_qtran"
    algo = "qplex"
    # algo = "qmix_maic"  # team modeling may not work in matrix games
    # algo = "qplex_maic"

    ### TODO Step2: choose matrix (for convenience for representation, we flatten the matrix into a vector)
    # ------- 2 player
    # payoff_flatten_vector= [1, 0, 0, 1]
    # payoff_flatten_vector=[8, 3, 2, -12, -13, -14, -12, -13, -14]
    # payoff_flatten_vector = [8, -12, -12, -12, 0, 0, -12, 0, 0]
    payoff_flatten_vector = [8, -12, -12, -12, 6, 0, -12, 0, 6]
    # payoff_flatten_vector = [20, 0, 0, 0, 12, 12, 0, 12, 12]
    # ------- 3 player
    # payoff_flatten_vector = [8, -12, -12, -12, 0, 0, -12, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0,
    #                          0]
    # ------- 4 player
    # payoff_flatten_vector = [8, -12, -12, -12, 0, 0, -12, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0,
    #                          0,
    #                          -12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                          -12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ### TODO Step3: choose other parameters, note that: action_num**agent_num = |payoff-matrix|
    action_num = 3
    agent_num = 2
    seed = 2  # fk, it is so tricky!
    round = 500
    hidden_num = 32

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train()
