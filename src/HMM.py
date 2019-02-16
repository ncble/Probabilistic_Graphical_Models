
import os, glob, sys
import numpy as np
class HMM(object):
    """
    A simple HMM model (discrete observation space). 
    Support D trajectories of observations (Obs.shape = (D, T)).
    
    init_distr: initial distribution of states shape = (num_s, )
    A: transition proba matrix shape = (num_s, num_s)
    B: emission proba matrix shape = (num_s, num_o)
    T: length of each trajectory
    
    num_s : number of states
    num_o : number of observation values

    """
    def __init__(self, init_distr, A, B, use_log=False):
        super(HMM, self).__init__()
        self.init_distr = init_distr ## shape = (num_s, )
        self.A = A ## shape = (num_s, num_s)
        self.B = B ## shape = (num_s, num_o)
        self.num_s, self.num_o = self.B.shape
        self.use_log = use_log
        if self.use_log:
            self.init_distr = np.log(init_distr)
            self.A = np.log(self.A)
            self.B = np.log(self.B)


    def compute_log_sum(self, X, axis=0):
        """
        Suppose that X = (log x1, log x2, ..., log xn) is already in log-scale.
        We want log(sum x_i). First, we substract by max (the most important term), 
        np.exp, np.sum and np.log.
        log x_i -> log(x_i/x_max) -> x_i/x_max -> sum x_i/x_max -> log(sum x_i/x_max)
          -> log(sum x_i/x_max) + log x_max

        keepdims is always true !
        """
        max_tempo = np.max(X, axis=axis, keepdims=True)
        return max_tempo + np.log(np.sum(np.exp(X-max_tempo), axis=axis, keepdims=True))
    def forward(self, Obs):
        """
        forward message alpha
        Obs: observations of shape (T, )

        return alpha, shape = (T, num_s)
        """
        if len(Obs.shape) == 1:
            ## only one trajectory
            T = len(Obs)
            alpha = np.zeros((T, self.num_s))
            if self.use_log:
                log_A = self.A
                log_B = self.B
                alpha[0] = self.init_distr + log_B[:, Obs[0]]
                for t in range(1, T):
                    tempo = alpha[t-1][:, None] + log_A # shape = (num_s, num_s)
                    alpha[t] = self.compute_log_sum(tempo, axis=0) + log_B[:, Obs[t]]
            else:
                alpha[0] = self.init_distr * self.B[:, Obs[0]]
                for i in range(1, T):
                    alpha[i] = alpha[i-1][None, :].dot(self.A)*self.B[:, Obs[i]]
        else:
            ## multi-trajectories
            D, T = Obs.shape
            alpha = np.zeros((D, T, self.num_s))
            if self.use_log:
                log_A = self.A
                log_B = self.B
                alpha[:, 0] = self.init_distr + log_B[:, Obs[:, 0]].T ## shape =(num_s, )+(D, num_s)=(D, num_s)
                for t in range(1, T):
                    tempo = alpha[:, t-1][:, :, None] + log_A # shape = (D, num_s, num_s)
                    alpha[:, t] = np.squeeze(self.compute_log_sum(tempo, axis=1)) + (log_B[:, Obs[:, t]].T)
            else:
                raise ValueError("Not implemented error.")
                alpha[0] = self.init_distr * self.B[:, Obs[0]]
                for i in range(1, T):
                    alpha[i] = alpha[i-1][None, :].dot(self.A)*self.B[:, Obs[i]]


        self.alpha = alpha
        return self.alpha
    def backward(self, Obs):
        """
        Obs: observations of shape (T, )

        return beta, shape = (T, num_s)
        """
        if len(Obs.shape) == 1:
            ## only one trajectory
            T = len(Obs)
            beta = np.zeros((T, self.num_s))
            if self.use_log:
                ### Since log(1) = 0, we don't need the initialization: "beta[-1] = np.ones(self.num_s)".
                log_A = self.A
                log_B = self.B
                for t in range(T-2, -1,-1):
                    tempo = (beta[t+1])[None, :] +  log_A + log_B[:, Obs[t+1]] # shape = (num_s, num_s)
                    beta[t] = np.squeeze(self.compute_log_sum(tempo, axis=1))
            else:
                beta[-1] = np.ones(self.num_s)
                for t in range(T-2, -1,-1):
                    beta[t] = self.A.dot(self.B[:, Obs[t+1]]*beta[t+1])

        else:
            ## multi-trajectories
            D, T = Obs.shape
            beta = np.zeros((D, T, self.num_s))
            if self.use_log:
                ### Since log(1) = 0, we don't need the initialization: "beta[-1] = np.ones(self.num_s)".
                log_A = self.A
                log_B = self.B
                for t in range(T-2, -1,-1):
                    tempo = (beta[:, t+1])[:, None, :] +  log_A + log_B[:, Obs[:, t+1]].T[:, None, :] # shape = (D, num_s, num_s)
                    beta[:, t] = np.squeeze(self.compute_log_sum(tempo, axis=-1), axis=-1)
            else:
                raise ValueError("Not implemented error.")
                beta[-1] = np.ones(self.num_s)
                for t in range(T-2, -1,-1):
                    beta[t] = self.A.dot(self.B[:, Obs[t+1]]*beta[t+1])

        self.beta = beta
        return self.beta
    def _assert_forward_backward_consistency(self, Obs):
        self.forward(Obs)
        self.backward(Obs)
        if not self.use_log:
            proba_obs_forward = np.sum(self.alpha[-1])
            proba_obs_backward = np.sum(self.init_distr*self.beta[0]*self.B[:, Obs[0]])
        else:
            proba_obs_forward = np.sum(np.exp(self.alpha[-1]))
            proba_obs_backward = np.sum(np.exp(self.init_distr+self.beta[0]+self.B[:, Obs[0]]))
        assert np.allclose(proba_obs_forward, proba_obs_backward), "Find forward: {}, backward: {}".format(proba_obs_forward, proba_obs_backward)
        return True
    def compute_gamma_zeta(self, Obs):
        """
        No "for loop" implementation for conditional proba !
        gamma: P(i_t = q_i| O, lambda), shape = (T, num_s)
        zeta : P(i_t = q_i, i_t+1 = q_j| O, lambda), shape = (T, num_s, num_s)

        """
        if len(Obs.shape) == 1:
            ## only one trajectory
            if self.use_log:
                gamma = self.alpha+self.beta ## shape = (T, num_s) 
                # max_state = np.max(gamma, axis=1, keepdims=True) ## shape = (T, 1)
                # denominator = max_state + np.log(np.sum(np.exp(gamma - max_state), axis=1, keepdims=True)) ## shape = (T,1)
                denominator = self.compute_log_sum(gamma, axis=1)
                gamma = gamma-denominator

                part1 = self.alpha[..., None]+self.A
                part2 = self.B.T[Obs[:],:]+self.beta ## shape = (T, num_s)
                
                part1 = part1[:-1] # shape = (T-1, num_s, num_s)
                part2 = part2[1:] # shape = (T-1, num_s)

                zeta = part1+part2[:, None, :]## shape = (T-1, num_s, num_s)
                # max_states = np.max(zeta, axis=(-2,-1), keepdims=True) ## shape (T-1, 1, 1)
                # denominator = max_states + np.log(np.sum(np.exp(zeta - max_states), axis=(1,2), keepdims=True))
                denominator = self.compute_log_sum(zeta, axis=(1,2))
                zeta = zeta-denominator ## shape = (T-1, num_s, num_s)
            else:
                gamma = self.alpha*self.beta ## shape = (T, num_s)
                gamma = gamma / np.sum(gamma, axis=1, keepdims=True) ## shape = (T, num_s)
                
                part1 = self.alpha[..., None]*self.A ## shape = (T, num_s, num_s), since (T, N, 1) * (N, N) = (T, N, N)
                part2 = self.B.T[Obs[:],:]*self.beta ## shape = (T, num_s)
                
                part1 = part1[:-1] # shape = (T-1, num_s, num_s)
                part2 = part2[1:] # shape = (T-1, num_s)

                zeta = part1*part2[:, None, :]## shape = (T-1, num_s, num_s)
                zeta = zeta/np.sum(zeta, axis=(1,2), keepdims=True) ## shape = (T-1, num_s, num_s)
        else:
            ## multi-trajectories
            if self.use_log:
                gamma = self.alpha+self.beta ## shape = (D, T, num_s) 
                # max_state = np.max(gamma, axis=1, keepdims=True) ## shape = (T, 1)
                # denominator = max_state + np.log(np.sum(np.exp(gamma - max_state), axis=1, keepdims=True)) ## shape = (T,1)
                denominator = self.compute_log_sum(gamma, axis=-1)
                gamma = gamma-denominator
                # assert np.allclose(np.sum(np.exp(gamma), axis=-1), np.ones(Obs.shape))
                part1 = self.alpha[..., None]+self.A ## shape = (D, T, num_s, num_s)
                part2 = self.B.T[Obs[:],:]+self.beta ## shape = (D, T, num_s)
                
                part1 = part1[:, :-1] # shape = (D, T-1, num_s, num_s)
                part2 = part2[:, 1:] # shape = (D, T-1, num_s)

                zeta = part1+part2[:, :, None, :]## shape = (D, T-1, num_s, num_s)
                # max_states = np.max(zeta, axis=(-2,-1), keepdims=True) ## shape (T-1, 1, 1)
                # denominator = max_states + np.log(np.sum(np.exp(zeta - max_states), axis=(1,2), keepdims=True))
                denominator = self.compute_log_sum(zeta, axis=(-1,-2))
                zeta = zeta-denominator ## shape = (T-1, num_s, num_s)
            else:
                raise ValueError("Not implemented error.")
                gamma = self.alpha*self.beta ## shape = (T, num_s)
                gamma = gamma / np.sum(gamma, axis=1, keepdims=True) ## shape = (T, num_s)
                
                part1 = self.alpha[..., None]*self.A ## shape = (T, num_s, num_s), since (T, N, 1) * (N, N) = (T, N, N)
                part2 = self.B.T[Obs[:],:]*self.beta ## shape = (T, num_s)
                
                part1 = part1[:-1] # shape = (T-1, num_s, num_s)
                part2 = part2[1:] # shape = (T-1, num_s)

                zeta = part1*part2[:, None, :]## shape = (T-1, num_s, num_s)
                zeta = zeta/np.sum(zeta, axis=(1,2), keepdims=True) ## shape = (T-1, num_s, num_s)

        self.gamma = gamma
        self.zeta = zeta
        return gamma, zeta

    def update_params(self, Obs):
        """
        Update parameters using forward/backward estimation: init_distr, A, B

        """

        if len(Obs.shape) == 1:
            ## only one trajectory
            if self.use_log:
                new_init_distr = self.gamma[0]
                part1 = np.squeeze(self.compute_log_sum(self.zeta, axis=0)) ## shape = (num_s, num_s)
                part2 = np.squeeze(self.compute_log_sum(self.gamma[:-1], axis=0))[:, None] ## shape (num_s, 1)
                new_A = part1 - part2

                tempo_index = Obs[:, None] == np.arange(self.num_o) ## shape = (T, num_o)
                denominator = self.compute_log_sum(self.gamma, axis=0)
                ## compute_log_sum for numerator
                numerator_max = np.max(self.gamma, axis=0) ## shape = (s, )
                numerator_ori = np.exp(self.gamma-numerator_max) ## shape = (T, s)
                numerator = numerator_ori.T.dot(tempo_index) ## shape = (s, o) # sum over index (Obs == k) and time T 
                numerator = np.log(numerator)+numerator_max[:, None] ## RuntimeWarning: divide by zero encountered in log

                new_B = numerator-np.squeeze(denominator)[:, None]
                
            else:

                new_init_distr = self.gamma[0]
                new_A = np.sum(self.zeta, axis=0)/(np.sum(self.gamma[:-1], axis=0)[:, None])

                tempo_index = Obs[:, None] == np.arange(self.num_o) ## shape = (T, num_o)
                new_B = self.gamma.T.dot(tempo_index)
                new_B = new_B/(np.sum(self.gamma, axis=0)[:, None])
        else:
            ## multi-trajectories
            if self.use_log:
                new_init_distr = np.squeeze(self.compute_log_sum(self.gamma[:, 0], axis=0)) - np.log(len(Obs))
                part1 = np.squeeze(self.compute_log_sum(self.zeta, axis=(1, 0)), axis=(1,0)) ## shape = (num_s, num_s)
                part2 = np.squeeze(self.compute_log_sum(self.gamma[:, :-1], axis=(1, 0)))[..., None] ## shape (num_s, 1)
                new_A = part1 - part2

                tempo_index = Obs[..., None] == np.arange(self.num_o) ## shape = (D, T, num_o)
                denominator = self.compute_log_sum(self.gamma, axis=(1,0)) ## shape = (1, 1, num_s)
                ## compute_log_sum for numerator
                numerator_max = np.max(self.gamma, axis=1) ## shape = (D, num_s)
                numerator_ori = np.exp(self.gamma-numerator_max[:, None, :]) ## shape = (D, T, num_s)
                
                numerator = np.sum(numerator_ori[..., None]*tempo_index[:,:,None,:], axis=1) ## shape = (D, num_s, num_o) # sum over index (Obs == k) and time T 

                numerator = np.log(numerator)+numerator_max[..., None] ## RuntimeWarning: divide by zero encountered in log

                new_B = np.squeeze(self.compute_log_sum(numerator, axis=0))-np.squeeze(denominator)[:, None]
                
            else:
                raise ValueError("Not implemented error.")
                new_init_distr = self.gamma[0]
                new_A = np.sum(self.zeta, axis=0)/(np.sum(self.gamma[:-1], axis=0)[:, None])

                tempo_index = Obs[:, None] == np.arange(self.num_o) ## shape = (T, num_o)
                new_B = self.gamma.T.dot(tempo_index)
                new_B = new_B/(np.sum(self.gamma, axis=0)[:, None])

        self.init_distr = new_init_distr
        self.A = new_A
        self.B = new_B
        return new_init_distr, new_A, new_B

    def generate_trajectories(self, length, traj=1, init_distr=None, A=None, B=None): # 
        """
        length: time T of trajectories
        init_distr.shape = (s,)
        A.shape = (s, s): transition proba
        B.shape = (s, o): emission proba
        traj: number of trajectories
        
        if init_distr=None, A=None, B=None, then use self.A,B,init_distr

        return observations.shape = (traj, length)
        """
        #     s = len(init_distr)
        if init_distr is None:
            if self.use_log:
                init_distr = np.exp(self.init_distr)
            else:
                init_distr = self.init_distr
        if A is None:
            if self.use_log:
                A = np.exp(self.A)
            else:
                A = self.A
        if B is None:
            if self.use_log:
                B = np.exp(self.B)
            else:
                B = self.B

        s, o = B.shape
        observations = []
        state = np.random.choice(np.arange(s), p=init_distr, size=(traj,))
        obs = random_choice_multidim(p=B[state], size=1)

        observations.append(obs[:, None])
        for i in range(1, length):
            state = random_choice_multidim(p=A[state], size=1)
            obs = random_choice_multidim(p=B[state], size=1)
            
            observations.append(obs[:, None])
        return np.concatenate(observations, axis=1)

    def main(self, iterations, Obs, verbose=0, A_gt=None, init_distr_gt=None):
        # print(np.sum(np.abs(A_gt-np.exp(self.A))))
        # print("="*10)
        for i in range(iterations):
            
            alpha = self.forward(Obs)
            beta = self.backward(Obs)
            self.compute_gamma_zeta(Obs)
            init_distr_es, A_es, B_es = self.update_params(Obs)
            if verbose:
                if self.use_log:
                    alpha = np.exp(alpha)
                    beta = np.exp(beta)
                    # init_distr = np.exp(self.init_distr)
                    # B = np.exp(self.B)
                    # A = np.exp(self.A)
                else:
                    alpha = alpha ## shape = (D, T, num_s)
                    beta = beta  ## shape = (D, T, num_s)
                    # init_distr = self.init_distr
                    # B = self.B
                    # A = self.A
                # forward_proba = np.mean(np.sum(alpha[:, -1], axis=-1))
                # backward_proba = np.sum(init_distr*B[:, Obs[0, 0]]*beta[0, 0])
                # print(forward_proba, backward_proba)
                # print(forward_proba)
                print(np.sum(np.abs(A_gt-np.exp(A_es))), np.sum(np.abs(init_distr_gt-np.exp(init_distr_es))))
                # print()
                # print(np.sum(np.abs(self.A-A_es)))
                
    def _sanity_check(self, use_log=True):
        self.use_log = use_log ## BAD, TODO

        init_distr = np.array([0.2,0.4,0.4])
        A = np.array([[0.5,0.2,0.3], [0.3,0.5,0.2], [0.2,0.3,0.5]])
        B = np.array([[0.5,0.5], [0.4,0.6], [0.7,0.3]])
        obs = np.array([0,1,0])

        alpha = self.forward(obs)
        beta = self.backward(obs)
        gamma, zeta = self.compute_gamma_zeta(obs)
        if use_log:
            alpha = np.exp(alpha)
            beta = np.exp(beta)
            gamma = np.exp(gamma)
            zeta = np.exp(zeta)
        else:
            alpha = alpha ## shape = (T, num_s)
            beta = beta  ## shape = (T, num_s)
            gamma = gamma ## shape = (T, num_s)
            zeta = zeta  ## shape = (T-1, num_s, num_s)
        print("The proba of observation sequence (using forward message): {}".format(np.sum(alpha[-1])))
        print("The proba of observation sequence (using backward message): {}".format(np.sum(init_distr*B[:, obs[0]]*beta[0])))
        print("The exact proba of observation sequence [0,1,0] is: 0.13022")
        print("Forward and backward are consistent: {}".format(self._assert_forward_backward_consistency(obs)))
        assert np.allclose(np.sum(gamma, axis=-1), np.ones(len(gamma))), "gamma is wrong (not normalized)"
        assert np.allclose(np.sum(zeta, axis=(-1,-2)), np.ones(len(zeta))), "zeta is wrong (not normalized)"
        assert np.allclose(np.sum(alpha[-1]), np.sum(init_distr*B[:, obs[0]]*beta[0])), "forward/backward inconsistent"

        alpha_gt = np.array([[0.1, 0.16, 0.28],
                             [0.077, 0.1104, 0.0606],
                             [0.04187, 0.035512, 0.052836]])
        beta_gt = np.array([[0.2451, 0.2622, 0.2277],
                             [0.54, 0.49, 0.57],
                             [1., 1., 1.]])
        assert np.allclose(alpha, alpha_gt), "alpha is wrong"
        assert np.allclose(beta, beta_gt), "beta is wrong"
        print('All passed!')
        return True

def random_choice_multidim(p=None, size=1):
    """
    Denote N the number of objects (discrete random choice).
    Given D distributions p.shape = (D, N), return samples.shape = (D, )
    e.g. D = 4, N = 3
    p = np.array([[0.2, 0.1, 0.7],
                  [0.3, 0.3, 0.4],
                  [0.1, 0.5, 0.4],
                  [0.7, 0.1, 0.2]])
    when p.shape = (N, ), this is equivalent to 
        np.random.choice(np.arange(N), p=p)
    
    return samples of shape = (D, ) + size = (D, size)
    """
    if len(p.shape) == 1:
        samples = np.random.choice(np.arange(len(p)), p=p, size=size)

    else:
        assert len(p.shape) == 2, "p.shape == (D, N); D distributions, N objects"
        assert type(size) == int, "only support int for now, this could be updated"
        D, N = p.shape
        # index = np.random.rand(D, *size) ### uniform sampling from [0, 1] of shape (D, )+size
        index = np.random.rand(D, size)
        cumsum_p = np.cumsum(p, axis=-1) ## TODO TOO WASTE ?
        if size>1:
            samples = np.max(np.cumsum(index[...,None]>cumsum_p[:, None, :], axis=-1), axis=-1)
        else:
            samples = np.max(np.cumsum(index>cumsum_p, axis=-1), axis=-1)
    return samples


if __name__=="__main__":
    print("Start")
    """
    ### Simple sample
    init_distr = np.array([0.2,0.4,0.4])
    A = np.array([[0.5,0.2,0.3], [0.3,0.5,0.2], [0.2,0.3,0.5]])
    B = np.array([[0.5,0.5], [0.4,0.6], [0.7,0.3]])
    obs = np.array([0,1,0]) # ,0,0,0,0,1,1,1,1,1
    obs = obs[None, :]
    algo = HMM(init_distr, A, B, use_log=True)
    alpha = algo.forward(obs)[0]
    beta = algo.backward(obs)[0]
    # alpha = algo.forward(obs)
    # beta = algo.backward(obs)
    # import ipdb; ipdb.set_trace()
    print(np.exp(alpha))
    print("The proba of observation sequence (using forward message): {}".format(np.sum(np.exp(alpha[-1]))))
    print("The proba of observation sequence (using backward message): {}".format(init_distr.dot(np.exp(beta)[0]*np.exp(algo.B)[:, obs[0,0]])))
    # print("The proba of observation sequence (using forward message): {}".format(np.sum(alpha[-1])))
    # print("The proba of observation sequence (using backward message): {}".format(init_distr.dot(beta[0]*algo.B[:, obs[0]])))
    print("The exact proba of observation sequence [0,1,0] is: 0.13022")
    # print("Forward and backward are consistent: {}".format(algo._assert_forward_backward_consistency(obs)))
    """
    # print("="*50)
    # print("Wiki sample")
    # print("="*20)
    # ### Wiki sample
    # init_distr = np.array([0.6,0.4])
    # A = np.array([[0.7,0.3], [0.4,0.6]])
    # B = np.array([[0.5,0.4,0.1], [0.1,0.3,0.6]])
    # # obs = np.array([0])
    # # obs = np.array([0,1])
    # obs = np.array([0,1,2])

    # algo = HMM(init_distr, A, B, use_log=True)
    # alpha = algo.forward(obs)
    # beta = algo.backward(obs)
    # print("Observation sequence: {}".format(obs))
    # print("The proba of observation sequence (using forward message): {}".format(np.sum(alpha[-1])))
    # print("The proba of observation sequence (using backward message): {}".format(init_distr.dot(beta[0]*algo.B[:, obs[0]])))
    # print("The exact proba of observation sequence [0] is: 0.4*0.1 + 0.6*0.5 = 0.34")
    # print("The exact proba of observation sequence [0, 1] is: 0.4*0.1*(0.6*0.3+0.4*0.4) + 0.6*0.5*(0.7*0.4+0.3*0.3)=0.1246")
    # print("The exact proba of observation sequence [0, 1, 2] is: 0.03628")
    # print("Forward and backward are consistent: {}".format(algo._assert_forward_backward_consistency(obs)))


    # obs = np.array([0,1,2,2,2,2,2])
    # algo = HMM(init_distr, A, B, use_log=False)
    # algo.main(10, obs)
    # print(algo.A)
    # print(algo.B)
    # algo = HMM(init_distr, A, B, use_log=True)
    # algo.main(10, obs)
    # print(np.exp(algo.A))
    # print(np.exp(algo.B))




    # print("="*50)
    # print("="*50)

    # init_distr = np.array([0.2,0.4,0.4])
    # A = np.array([[0.5,0.2,0.3], [0.3,0.5,0.2], [0.2,0.3,0.5]])
    # B = np.array([[0.5,0.5], [0.4,0.6], [0.7,0.3]])
    # obs = np.array([0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0]) # ,0,0,0,0,1,1,1,1,1

    # algo = HMM(init_distr, A, B, use_log=False)
    # alpha = algo.forward(obs)
    # beta = algo.backward(obs)
    # print(np.sum(alpha[-1]))
    # print(np.sum(np.exp(alpha[-1])))
    # ===================================
    ### Multi trajectories test
    init_distr_gt = np.array([0.2,0.4,0.4])
    A_gt = np.array([[0.5,0.2,0.3], [0.3,0.5,0.2], [0.2,0.3,0.5]])
    B = np.array([[0.5,0.5], [0.4,0.6], [0.7,0.3]])
    algo = HMM(init_distr_gt, A_gt, B, use_log=True)
    data = algo.generate_trajectories(10, traj=500)
    # algo._sanity_check()
    noise = 1.0
    A = A_gt + noise*np.random.random((3,3))
    A = A/np.sum(A, axis=1, keepdims=True)
    B = B + noise*np.random.random((3, 2))
    B = B/np.sum(B, axis=1, keepdims=True)
    init_distr = init_distr_gt + noise*np.random.random(3)
    init_distr = init_distr/np.sum(init_distr)
    algo = HMM(init_distr, A, B, use_log=True)
    # import ipdb; ipdb.set_trace()
    # algo.forward(data)
    algo.main(50, data, A_gt=A_gt, init_distr_gt=init_distr_gt)
    print(np.exp(algo.init_distr))
    print(np.exp(algo.A))
    print(np.exp(algo.B))

    # proba = np.sum(np.exp(self.alpha[:,-1,:]), axis=-1)


