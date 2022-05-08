# -*- coding:UTF-8 -*-
# @Time    : 2021/7/19 16:13
# @Author  : HaoXiaotian
# @Contact : xiaotianhao@tju.edu.cn
# @File    : q_family.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

        # (agent_num, action_num)
        self.individual_qs = Parameter(torch.randn(1, agent_num, action_num))

        if algo in ["vdn", "vdn_qtran"]:
            pass
        elif algo in "weighted_vdn":
            self.w1 = Parameter(torch.randn([agent_num, 1], dtype=torch.float32))
            self.b1 = Parameter(torch.randn([1, 1], dtype=torch.float32))
        elif algo in ["qmix", "qmix_qtran"]:
            # (agent_num, embedding_dim)
            self.w1 = Parameter(torch.randn([agent_num, hidden_num], dtype=torch.float32))
            self.b1 = Parameter(torch.randn([1, hidden_num], dtype=torch.float32))
            # (embedding_dim, 1)
            self.w2 = Parameter(torch.randn([hidden_num, 1], dtype=torch.float32))
            self.b2 = Parameter(torch.randn([1, 1], dtype=torch.float32))
        elif algo == "qplex":
            # lambda net
            self.lambda_net = nn.Sequential(
                nn.Linear(agent_num * action_num, hidden_num),
                nn.ReLU(),
                nn.Linear(hidden_num, agent_num),
            )
        else:
            raise NotImplementedError

        # init parameters
        self._init_parameters()

    def _init_parameters(self):
        # init individual q-value
        init.kaiming_uniform_(self.individual_qs, a=math.sqrt(5))
        # init.uniform_(self.individual_qs, 0, 0)
        print("******************* [q_i] init q tables *******************")
        q_print = self.individual_qs[0]  # [agent_num, action_num]
        for agent_idx in range(self.agent_num):
            individual_q = q_print[agent_idx]  # [action_num]
            print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                    individual_q.max(
                                                                                        dim=0)[1].item()))
            print(individual_q.tolist())
            print("--------------------------------------\n")

    def forward(self, batch_size, batch_action, q_joint, print_log=False):
        """
        :param batch_size: 4
        :param batch_action: [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]
        :return:
        """
        assert batch_action.shape[0] == batch_size
        assert batch_action.shape[1] == self.agent_num
        assert batch_action.shape[2] == 1
        # Get action values (batch_size, agent_num)
        selected_individual_qs = torch.gather(self.individual_qs.expand(batch_size, self.agent_num, self.action_num),
                                              dim=2, index=batch_action.long()).squeeze(2)  # Remove the last dim

        # Calculate q_total
        if self.algo in ["vdn", "vdn_qtran"]:
            q_tot = selected_individual_qs.sum(dim=1, keepdim=True)
        elif self.algo == "weighted_vdn":
            q_tot = torch.mm(selected_individual_qs, torch.abs(self.w1)) + self.b1
        elif self.algo in ["qmix", "qmix_qtran"]:
            hidden = F.elu(torch.mm(selected_individual_qs, torch.abs(self.w1)) + self.b1)
            q_tot = torch.mm(hidden, torch.abs(self.w2)) + self.b2
        elif self.algo == "qplex":
            q_upper, _ = self.individual_qs.max(dim=2, keepdim=False)
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
            q_print = self.individual_qs[0]  # [agent_num, action_num]
            for agent_idx in range(self.agent_num):
                individual_q = q_print[agent_idx]  # [action_num]
                print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                        individual_q.max(
                                                                                            dim=0)[1].item()))
                print(individual_q.tolist())
                print("--------------------------------------\n")

        # Calculate loss
        if self.algo in ["vdn_qtran", "qmix_qtran"]:
            # greedy actions
            individual_greedy_action = self.individual_qs.max(dim=2, keepdim=True)[1]
            # the sample in current batch in which the actions == greedy actions
            max_point_mask = ((individual_greedy_action == batch_action).long().sum(axis=1) == self.agent_num).float()
            q_clip = torch.max(q_tot, q_joint).detach()  # Qtran核心： to ensure q_tot >= q_label
            loss = torch.mean(
                max_point_mask * ((q_tot - q_joint) ** 2) + (1 - max_point_mask) * ((q_tot - q_clip) ** 2))
        else:
            loss = torch.mean((q_tot - q_joint) ** 2)
        return q_tot, loss


def train(case=0):
    """
    [
      [1, 0],
      [0, 1]
    ]
    :return:
    """
    agent_num = 2
    if case == 0:
        batch_size = 4
        action_num = 2
        batch_action = torch.from_numpy(
            np.array([
                [[0], [0]],
                [[0], [1]],
                [[1], [0]],
                [[1], [1]]
            ], dtype=np.compat.long)
        )
        q_joint = torch.from_numpy(np.reshape(
            np.array([1, 0, 0, 1], dtype=np.float32)
            , newshape=[batch_size, 1])
        )
    elif case == 1:
        batch_size = 9
        action_num = 3
        batch_action = torch.from_numpy(
            np.array([
                [[0], [0]],
                [[0], [1]],
                [[0], [2]],
                [[1], [0]],
                [[1], [1]],
                [[1], [2]],
                [[2], [0]],
                [[2], [1]],
                [[2], [2]],
            ], dtype=np.compat.long)
        )
        q_joint = torch.from_numpy(np.reshape(
            np.array([8, 3, 2, -12, -13, -14, -12, -13, -14], dtype=np.float32)
            , newshape=[batch_size, 1])
        )

    elif case == 2:
        batch_size = 9
        action_num = 3
        batch_action = torch.from_numpy(
            np.array([
                [[0], [0]],
                [[0], [1]],
                [[0], [2]],
                [[1], [0]],
                [[1], [1]],
                [[1], [2]],
                [[2], [0]],
                [[2], [1]],
                [[2], [2]],
            ], dtype=np.compat.long)
        )
        q_joint = torch.from_numpy(np.reshape(
            np.array([8, -12, -12, -12, 0, 0, -12, 0, 0], dtype=np.float32)
            , newshape=[batch_size, 1])
        )

    elif case == 3:
        batch_size = 9
        action_num = 3
        batch_action = torch.from_numpy(
            np.array([
                [[0], [0]],
                [[0], [1]],
                [[0], [2]],
                [[1], [0]],
                [[1], [1]],
                [[1], [2]],
                [[2], [0]],
                [[2], [1]],
                [[2], [2]],
            ], dtype=np.compat.long)
        )
        q_joint = torch.from_numpy(np.reshape(
            np.array([8, -12, -12, -12, 6, 0, -12, 0, 6], dtype=np.float32)
            , newshape=[batch_size, 1])
        )
    elif case == 4:
        batch_size = 9
        action_num = 3
        batch_action = torch.from_numpy(
            np.array([
                [[0], [0]],
                [[0], [1]],
                [[0], [2]],
                [[1], [0]],
                [[1], [1]],
                [[1], [2]],
                [[2], [0]],
                [[2], [1]],
                [[2], [2]],
            ], dtype=np.compat.long)
        )
        q_joint = torch.from_numpy(np.reshape(
            np.array([8, -12, -12, -12, 0, 0, -12, 0, 0], dtype=np.float32) + 12
            , newshape=[batch_size, 1])
        )
    else:
        raise NotImplementedError

    one_step_q_network = QFamily(algo=algo, agent_num=agent_num, action_num=action_num, hidden_num=hidden_num)
    optimizer = torch.optim.Adam(params=one_step_q_network.parameters(), lr=0.01)

    for epoch in range(round):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part I: non-monotonic QMIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        q_tot, loss = one_step_q_network.forward(batch_size, batch_action, q_joint)
        if epoch % 100 == 0:
            print("Iter={}: MSE loss={}".format(epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    q_tot, _ = one_step_q_network.forward(batch_size, batch_action, q_joint, print_log=True)

    print("******************* Predicted Q_tot: *******************")
    q_print = q_tot.detach().tolist()
    for row in range(action_num):
        start_pos = row * action_num
        print(q_print[start_pos: start_pos + action_num])

    print("******************* True Q_joint: *******************")
    q_print = q_joint.detach().tolist()
    for row in range(action_num):
        start_pos = row * action_num
        print(q_print[start_pos: start_pos + action_num])


if __name__ == "__main__":
    # algo = "vdn"
    # algo = "weighted_vdn"
    # algo = "qmix"
    # algo = "vdn_qtran"
    # algo = "qmix_qtran"
    algo = "qplex"

    round = 4000
    hidden_num = 2

    train(case=3)
