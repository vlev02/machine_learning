"""HMM模型的计算方法"""
import numpy as np


class HmmModel:
    """HMM模型的计算方法类"""
    @staticmethod
    def forward_prob_calculate(pi, mat_a, mat_b, o_li):
        """概率计算问题——前向方法
        :params
            pi: 初始状态概率向量
            mat_a: 状态转移矩阵
            mat_b: 观测概率矩阵
            o_li: 观测序列
        :return
            prob: 概率值
            alp_li: alpha_t
        """
        # TODO: params format check
        n_s = pi.shape[0]
        n_o = mat_b.shape[1]
        alp_li = []
        # alpha_t[j] = p(o1, o2, ... , ot, st=j)
        alpha_t = pi * mat_b[:, o_li[0]]
        alp_li.append(alpha_t)
        for o in o_li[1:]:
            # alpha_(t+1)[j] = \sum_i alpha_t[i] * a_(i,j) * b_(j,ot)
            alpha_t = (alpha_t @ mat_a) * mat_b[:, o]
            alp_li.append(alpha_t)

        prob = np.sum(alpha_t)
        alp_li = np.array(alp_li)
        return prob, alp_li

    @staticmethod
    def backward_prob_calculate(pi, mat_a, mat_b, o_li):
        """概率计算问题——前向方法
        :params
            pi: <np.ndarray[n_s]> 初始状态概率向量
            mat_a: <np.ndarray[n_o * n_o]> 状态转移矩阵
            mat_b: <np.ndarray[n_o * n_s]> 观测概率矩阵
            o_li: <list> 观测序列
        :return
            prob: 概率值
        """
        # TODO: params format check
        n_s = pi.shape[0]
        n_o = mat_b.shape[1]
        bet_li = []

        # beta_t[i] = p(o_t+1, o_t+2, ... , o_T | s_t=i)
        beta_t = np.ones(n_s)
        bet_li.append(beta_t)
        for o in o_li[1:][::-1]:
            # beta_(t-1)[i] = \sum_j a_(i, j) * b_(j, o_t) * beta_t(j)
            beta_t = mat_a @ (mat_b[:, o] * beta_t)
            bet_li.append(beta_t)

        bet_li = np.array(bet_li[::-1])
        prob = np.sum(pi * mat_b[:, o_li[0]] * beta_t)
        return prob, bet_li

    @classmethod
    def gamma_li(cls, pi, mat_a, mat_b, o_li):
        """给定模型参数和观测序列，计算t时刻状态的概率分布
        :return
            gamma_li = gamma_1, gamma_2, ... , gamma_T. where gamma_t[i] = p(s_t=i | o_li)
        """
        gamma_li = []
        _, alp_li = cls.forward_prob_calculate(pi, mat_a, mat_b, o_li)
        _, bet_li = cls.backward_prob_calculate(pi, mat_a, mat_b, o_li)
        for alpha_t, beta_t in zip(alp_li, bet_li):
            p_li = alpha_t * beta_t
            p_li /= sum(p_li)
            gamma_li.append(p_li)
        return gamma_li

    @classmethod
    def baum_welch(cls, n_s, n_o, os_li, pi0=None, mat_a0=None, mat_b0=None):
        """EM算法学习hmm模型参数
        :param
            n_s: <int> 隐状态数量
            n_o: <int> 观测状态数量
            os_li: <list[list, ...]> 观测序列集
            pi0: <np.ndarray[n_s]> or None 初始状态概率向量的初始值
            mat_a0: <np.ndarray[n_o * n_o]> or None 状态转移矩阵的初始值
            mat_b0: <np.ndarray[n_o * n_s]> or None 观测概率矩阵的初始值
        :returns
            pi0: <np.ndarray[n_s]> or None 初始状态概率向量
            mat_a0: <np.ndarray[n_o * n_o]> or None 状态转移矩阵
            mat_b0: <np.ndarray[n_o * n_s]> or None 观测概率矩阵
        """
        # TODO: 提供hmm模型参数初始值的默认给定方法（if pi0 is None ...）
        if pi0 is None:
            pi0 = np.random.rand(n_s)
            pi0 /= pi0.sum()
        if mat_a0 is None:
            mat_a0 = np.random.rand(n_s, n_s)
            mat_a0 = np.diag(1/mat_a0.sum(axis=1)) @ mat_a0
        if mat_b0 is None:
            mat_b0 = np.random.rand(n_s, n_o)
            mat_b0 = np.diag(1/mat_b0.sum(axis=1)) @ mat_b0
        n_iter = 20  # EM迭代次数
        for i_iter in range(n_iter):
            p_t0 = np.zeros(n_s)
            p_tl_t = np.zeros([n_s, n_s])
            p_s_o = np.zeros([n_s, n_o])

            for o_li in os_li:
                _, alp_li = cls.forward_prob_calculate(pi0, mat_a0, mat_b0, o_li)
                _, bet_li = cls.backward_prob_calculate(pi0, mat_a0, mat_b0, o_li)
                first_flag = True
                for o_t, alpha_t, beta_t in zip(o_li, alp_li, bet_li):
                    pli_t = alpha_t * beta_t  # pli_t[i] = p(O, s_t=i, pi0, mat_a0, mat_b0)
                    pli_tc = pli_t / pli_t.sum()  # pli_tc[i] = p(s_t=i | O, pi0, mat_a0, mat_b0)
                    if first_flag:
                        first_flag = False
                        p_t0 += pli_tc
                    else:
                        # pli_tl_t[i, j] = p(O, s_tl=i, s_t=j, pi0, mat_a0, mat_b0)
                        pli_tl_t = np.diag(alpha_tl) @ mat_a0 @ np.diag(mat_b0[:, o_t] * beta_t)
                        pli_tl_tc = pli_tl_t / pli_tl_t.sum()  # pli_tl_tc[i, j] = p(s_tl=i, s_t=j | O, pi0, mat_a0, mat_b0)
                        p_tl_t += pli_tl_tc
                    p_s_o[:, o_t] += pli_tc
                    alpha_tl = alpha_t

            pi0 = p_t0 / p_t0.sum()
            mat_a0 = np.diag(1 / p_tl_t.sum(axis=1)) @ p_tl_t
            mat_b0 = np.diag(1 / p_s_o.sum(axis=1)) @ p_s_o
        return pi0, mat_a0, mat_b0

    @staticmethod
    def viterbi_algorithm(pi, mat_a, mat_b, o_li):
        n_s = len(pi)
        u_tl = mat_b[:, o_li[-1]]
        v_tl = [[i] for i in range(n_s)]
        for o_t in o_li[:-1][::-1]:
            u_t_mat = np.diag(mat_b[:, o_t]) @ mat_a @ np.diag(u_tl)
            best_st = np.argmax(u_t_mat, axis=1)
            v_t = [[i] + v_tl[best_j] for i, best_j in enumerate(best_st)]
            # update
            u_tl = u_t_mat.max(axis=1)
            v_tl = v_t
        u_tl *= pi
        best_s0 = np.argmax(u_tl, axis=0)
        best_sli = v_tl[best_s0]
        best_prob = u_tl[best_s0]
        return best_sli, best_prob

if __name__ == "__main__":
    pi_ = np.array([.2, .4, .4])
    mat_a_ = np.array([[.5, .2, .3], [.3, .5, .2], [.2, .3, .5]])
    mat_b_ = np.array([[.5, .5], [.4, .6], [.7, .3]])
    o_li_ = [0, 1, 0]
    n_s_ = len(pi_)
    n_o_ = mat_b_.shape[1]

    hm = HmmModel()

    # forward_prob_calculate
    prob_, alp_li_ = hm.forward_prob_calculate(pi_, mat_a_, mat_b_, o_li_)
    print(prob_)

    # backward_prob_calculate
    prob_, bet_li_ = hm.backward_prob_calculate(pi_, mat_a_, mat_b_, o_li_)
    print(prob_)

    # gamma_li
    gamma_li_ = hm.gamma_li(pi_, mat_a_, mat_b_, o_li_)
    print(*gamma_li_)

    # baum_welch
    os_li_ = [[0, 1, 0], [1, 0, 1, 1, 0, 1, 0, 0, 0, 0]]
    hm.baum_welch(n_s_, n_o_, os_li=os_li_)

    # viterbi_algorithm
    best_sli_, best_prob_ = hm.viterbi_algorithm(pi_, mat_a_, mat_b_, o_li=[0, 1, 0, 1])
    print(best_sli_, best_prob_)
    pass
