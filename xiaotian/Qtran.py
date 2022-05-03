# -*- coding:UTF-8 -*-
# @Time    : 2021/7/18 16:13
# @Author  : HaoXiaotian
# @Contact : xiaotianhao@tju.edu.cn
# @File    : single_state_qmix.py


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math


class Qtran(nn.Module):
    def __init__(self, agent_num, action_num):
        super(Qtran, self).__init__()
        self.agent_num = agent_num
        self.action_num = action_num

        # (agent_num, action_num)
        self.individual_qs = Parameter(th.Tensor(1, agent_num, action_num))

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

    def forward(self, batch_size, batch_action, q_label, print_log=False):
        """
        :param batch_size: 4
        :param batch_action: [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]
        :return:
        """
        assert batch_action.shape[0] == batch_size
        assert batch_action.shape[1] == self.agent_num
        assert batch_action.shape[2] == 1

        # (batch_size, agent_num)
        selected_individual_qs = th.gather(self.individual_qs.expand(batch_size, self.agent_num, self.action_num),
                                           dim=2, index=batch_action.long()).squeeze(2)  # Remove the last dim

        q_tot = selected_individual_qs.sum(axis=1, keepdims=True)

        if print_log:
            print("******************* [q_i] Individual q tables *******************")
            q_print = self.individual_qs[0]  # [agent_num, action_num]
            for agent_idx in range(self.agent_num):
                individual_q = q_print[agent_idx]  # [action_num]
                print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                        individual_q.max(
                                                                                            dim=0)[1].item()))
                print(individual_q.tolist())
                print("--------------------------------------\n")

        # (1) VDN loss
        # loss = th.mean((q_tot - q_label) ** 2)

        # (2) QTran loss
        # greedy actions
        individual_greedy_action = self.individual_qs.max(dim=2, keepdim=True)[1]
        # the sample in current batch in which the actions == greedy actions
        max_point_mask = ((individual_greedy_action == batch_action).long().sum(axis=1) == self.agent_num).float()
        q_clip = th.max(q_tot, q_label).detach()  # to ensure q_tot >= q_label
        loss = th.mean(max_point_mask * ((q_tot - q_label) ** 2) + (1 - max_point_mask) * ((q_tot - q_clip) ** 2))

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
        batch_action = th.from_numpy(
            np.array([
                [[0], [0]],
                [[0], [1]],
                [[1], [0]],
                [[1], [1]]
            ], dtype=np.long)
        )
        q_joint = th.from_numpy(np.reshape(
            np.array([1, 0, 0, 1], dtype=np.float32)
            , newshape=[batch_size, 1])
        )
    elif case == 1:
        batch_size = 9
        action_num = 3
        batch_action = th.from_numpy(
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
            ], dtype=np.long)
        )
        q_joint = th.from_numpy(np.reshape(
            np.array([8, -12, -12, -12, 0, 0, -12, 0, 0], dtype=np.float32)
            , newshape=[batch_size, 1])
        )
    elif case == 2:
        batch_size = 9
        action_num = 3
        batch_action = th.from_numpy(
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
            ], dtype=np.long)
        )
        q_joint = th.from_numpy(np.reshape(
            np.array([8, 3, 2, -12, -13, -14, -12, -13, -14], dtype=np.float32)
            , newshape=[batch_size, 1])
        )
    elif case == 3:
        batch_size = 9
        action_num = 3
        batch_action = th.from_numpy(
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
            ], dtype=np.long)
        )
        q_joint = th.from_numpy(np.reshape(
            np.array([8, -12, -12, -12, 6, 0, -12, 0, 6], dtype=np.float32)
            , newshape=[batch_size, 1])
        )
    elif case == 4:
        batch_size = 9
        action_num = 3
        batch_action = th.from_numpy(
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
            ], dtype=np.long)
        )
        q_joint = th.from_numpy(np.reshape(
            np.array([8, -12, -12, -12, 0, 0, -12, 0, 0], dtype=np.float32) + 12
            , newshape=[batch_size, 1])
        )
    else:
        raise NotImplementedError

    qtran = Qtran(agent_num=agent_num, action_num=action_num)

    # optimizer = th.optim.RMSprop(params=qtran.parameters(), lr=0.01, alpha=0.99, eps=0.00001)
    # optimizer = th.optim.SGD(params=qtran.parameters(), lr=0.01)
    optimizer = th.optim.Adam(params=qtran.parameters(), lr=0.01)

    for epoch in range(round):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part I: non-monotonic QMIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        q_tot, loss = qtran.forward(batch_size, batch_action, q_joint)
        if epoch % 100 == 0:
            print("Iter={}: QTRAN loss={}".format(epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    q_tot, _ = qtran.forward(batch_size, batch_action, q_joint, print_log=True)

    print("******************* Predicted Q_tot: *******************")
    q_print = q_tot.detach().tolist()
    for row in range(action_num):
        start_pos = row * action_num
        print(q_print[start_pos: start_pos + action_num])
    print()

    print("******************* True Q_joint: *******************")
    q_print = q_joint.detach().tolist()
    for row in range(action_num):
        start_pos = row * action_num
        print(q_print[start_pos: start_pos + action_num])


if __name__ == "__main__":
    # monotonic
    seed = 12345  # top left
    # seed = 12  # down right
    # seed = 123456  # all 0.5
    round = 2000

    # non-monotonic
    # seed = 1  # true decomposition
    # seed = 2  # all 0.5
    # random.seed(seed)
    # np.random.seed(seed)
    # th.manual_seed(seed)
    # random_sample = True
    # random_sample = False


    # non monotonic
    # train(case=1)

    # monotonic
    train(case=1)