# -*- coding:UTF-8 -*-
# @Time    : 2020/12/11 10:20
# @Author  : HaoXiaotian
# @Contact : xiaotianhao@tju.edu.cn
# @File    : single_state_qmix.py


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class SingleStateQMIX(nn.Module):
    def __init__(self, algo: ["vdn", "qmix"], agent_num, action_num, embedding_dim, monotonic):
        super(SingleStateQMIX, self).__init__()
        self.agent_num = agent_num
        self.action_num = action_num
        self.embedding_dim = embedding_dim

        self.algo = algo
        self.monotonic = monotonic

        # (agent_num, action_num)
        self.individual_qs = Variable(th.randn([1, agent_num, action_num], dtype=th.float32), requires_grad=True)

        if self.algo == "vdn":
            pass
        elif self.algo == "weighted_vdn":
            self.w1 = Variable(th.randn([agent_num, 1], dtype=th.float32), requires_grad=True)
            self.b1 = Variable(th.randn([1, 1], dtype=th.float32), requires_grad=True)
        elif self.algo == "qmix":
            # (agent_num, embedding_dim)
            self.w1 = Variable(th.randn([agent_num, embedding_dim], dtype=th.float32), requires_grad=True)
            self.b1 = Variable(th.randn([1, embedding_dim], dtype=th.float32), requires_grad=True)

            # (embedding_dim, 1)
            self.w2 = Variable(th.randn([embedding_dim, 1], dtype=th.float32), requires_grad=True)
            self.b2 = Variable(th.randn([1, 1], dtype=th.float32), requires_grad=True)
        else:
            raise NotImplementedError

    def parameters(self, recurse=True):
        if self.algo == "vdn":
            return [self.individual_qs]
        elif self.algo == "weighted_vdn":
            return [self.individual_qs, self.w1, self.b1]
        elif self.algo == "qmix":
            return [self.individual_qs, self.w1, self.b1, self.w2, self.b2]

    def forward(self, batch_size, batch_action, print_log=False):
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

        if self.algo == "vdn":
            q_tot = selected_individual_qs.sum(dim=1, keepdim=True)

        elif self.algo == "weighted_vdn":
            if self.monotonic:
                q_tot = th.mm(selected_individual_qs, th.abs(self.w1)) + self.b1
            else:
                q_tot = th.mm(selected_individual_qs, self.w1) + self.b1

        elif self.algo == "qmix":
            # First layer
            # (batch_size, agent_num) * (agent_num, embedding_dim) = (batch_size, embedding_dim) + (1, embedding_dim)
            if self.monotonic:
                hidden = F.elu(th.mm(selected_individual_qs, th.abs(self.w1)) + self.b1)
            else:
                hidden = F.elu(th.mm(selected_individual_qs, self.w1) + self.b1)

            # Second layer
            # (batch_size, embedding_dim) * (embedding_dim, 1) = (batch_size, 1) + (1, 1)
            if self.monotonic:
                q_tot = th.mm(hidden, th.abs(self.w2)) + self.b2
            else:
                q_tot = th.mm(hidden, self.w2) + self.b2

        if print_log:
            print("******************* Individual q tables *******************")
            q_print = self.individual_qs[0]  # [agent_num, action_num]
            for agent_idx in range(self.agent_num):
                individual_q = q_print[agent_idx]  # [action_num]
                print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                        individual_q.max(
                                                                                            dim=0)[1].item()))
                print(individual_q.tolist())
                print("--------------------------------------")

        return q_tot


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
    else:
        raise NotImplementedError

    qmix = SingleStateQMIX(algo=algo, agent_num=agent_num, action_num=action_num, embedding_dim=agent_num,
                           monotonic=monotonic)
    optimizer = th.optim.RMSprop(params=qmix.parameters(), lr=0.01, alpha=0.99, eps=0.00001)
    # optimizer = th.optim.SGD(params=qmix.parameters(), lr=0.1)

    for _ in range(2000):
        q_tot = qmix.forward(batch_size, batch_action)
        loss = th.mean((q_tot - q_joint) ** 2)
        if _ % 100 == 0:
            print("Iter={}: MSE loss={}".format(_, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    q_tot = qmix.forward(batch_size, batch_action, print_log=True)

    print("******************* Predicted Q_tot: *******************")
    q_print = q_tot.detach().tolist()
    for row in range(action_num):
        start_pos = row * action_num
        print(q_print[start_pos: start_pos + action_num])

    print("Column difference:")
    q_print_2d = np.asarray(q_print).reshape([action_num, action_num])
    print(q_print_2d[:, 1] - q_print_2d[:, 0])
    print(q_print_2d[:, 2] - q_print_2d[:, 1])

    print("******************* True Q_joint: *******************")
    q_print = q_joint.detach().tolist()
    for row in range(action_num):
        start_pos = row * action_num
        print(q_print[start_pos: start_pos + action_num])


if __name__ == "__main__":
    # monotonic
    # seed = 12345  # top left
    # seed = 12  # down right
    # seed = 123456  # all 0.5

    # non-monotonic
    # seed = 1  # true decomposition
    # seed = 2  # all 0.5
    # random.seed(seed)
    # np.random.seed(seed)
    # th.manual_seed(seed)
    # random_sample = True
    # random_sample = False
    # monotonic = False


    algo = "vdn"
    # algo = "weighted_vdn"
    # algo = "qmix"
    monotonic = True
    # non monotonic
    # train(case=1)

    # monotonic
    train(case=2)
