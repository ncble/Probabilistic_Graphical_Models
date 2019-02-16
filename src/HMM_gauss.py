
import os, glob, sys
import numpy as np
class HMM_gaussian(object):
    """
    num_s : number of states
    num_o : number of observation values

    """
    def __init__(self, init_distr, A, B_mu, B_sigma, use_log=False):
        super(HMM_gaussian, self).__init__()
        self.init_distr = init_distr ## shape = (num_s, )
        self.A = A ## shape = (num_s, num_s)
        self.B_mu = B_mu ## shape = (num_s, dim)
        self.B_sigma = B_sigma ## shape = (num_s, dim, dim)
        
        self.num_s = len(self.A)
        # self.num_s, self.num_o = self.B.shape
        self.use_log = use_log
    def multivariate_gaussian_pdf(self, X, mu_all, sigma_all, log_scale=False):
        """
        X.shape = (T, dim)
        mu_all.shape = (num_s, dim,)
        sigma_all.shape = (num_s, dim, dim)

        return shape = (num_s, T)

        """
        dim = X.shape[-1]
        sigma_invs = np.stack([np.linalg.inv(sigma) for sigma in sigma_all]) ## shape = (num_s, dim, dim)
        sigma_det = np.array([np.linalg.det(sigma) for sigma in sigma_all]) ##  shape = (num_s, )
        

        if log_scale:
            factor = -0.5*(np.log((2*np.pi)**dim) + np.log(sigma_det))
            centered_part = X - np.expand_dims(mu_all, axis=1) ## shape (T, dim) - (num_s, 1, dim) = (num_s, T, dim)
            dot_part = np.sum(centered_part[..., None] * np.expand_dims(sigma_invs, axis=1), axis=-2) ## shape = (num_s, T, dim)
            all_part = np.sum(dot_part*centered_part, axis=-1) ## shape = (num_s, T)
            main_part = -0.5*all_part
            # import ipdb; ipdb.set_trace()
            return main_part + factor[:, None]

        else:
            factor = 1/np.sqrt((2*np.pi)**dim * sigma_det) ## shape = (num_s, )
            centered_part = X - np.expand_dims(mu_all, axis=1) ## shape (T, dim) - (num_s, 1, dim) = (num_s, T, dim)
            dot_part = np.sum(centered_part[..., None] * np.expand_dims(sigma_invs, axis=1), axis=-2) ## shape = (num_s, T, dim)
            all_part = np.sum(dot_part*centered_part, axis=-1) ## shape = (num_s, T)
            main_part = np.exp(-0.5*all_part)
            
            return  factor[:, None] * main_part 
    
    def build_emission_proba(self, Obs, log_scale=False):
        self.B = self.multivariate_gaussian_pdf(Obs, self.B_mu, self.B_sigma, log_scale=log_scale) ## shape = (num_s, T)

        return self.B
    def forward(self, Obs):
        """
        return alpha, shape = (T, num_s)
        """
        T = len(Obs)
        alpha = np.zeros((T, self.num_s))

        if self.use_log:
            log_A = self.A
            log_B = self.B
            alpha[0] = self.init_distr + log_B[:, 0]
            for t in range(1, T):
                alpha[t] = self.compute_log_sum(alpha[t-1][:, None] + log_A, axis=0) + log_B[:, t]
        else:
            alpha[0] = self.init_distr * self.B[:, 0]
            for t in range(1, T):
                alpha[t] = alpha[t-1][None, :].dot(self.A)*self.B[:, t]
        
        self.alpha = alpha
        return self.alpha
    def backward(self, Obs):
        """
        return beta, shape = (T, num_s)
        """
        T = len(Obs)
        beta = np.zeros((T, self.num_s))
        if self.use_log:
            ### Since log(1) = 0, we don't need the initialization: "beta[-1] = np.ones(self.num_s)".
            log_A = self.A
            log_B = self.B
            for t in range(T-2, -1,-1):
                beta[t] = np.squeeze(self.compute_log_sum((beta[t+1])[None, :] +  log_A + log_B[:, t+1], axis=1))


            # self.beta = np.exp(beta)
        else:
            beta[-1] = np.ones(self.num_s)
            for t in range(T-2, -1,-1):
                beta[t] = self.A.dot(self.B[:, t+1]*beta[t+1])

        self.beta = beta
        return self.beta
    def _assert_forward_backward_consistency(self, Obs):
        self.forward(Obs)
        self.backward(Obs)
        if self.use_log:
            assert 1==2
            proba_obs_forward = np.sum(np.exp(self.alpha)[-1])
            proba_obs_backward = np.sum(self.init_distr*np.exp(self.beta)[0]*self.B[:, 0])
        else:
            proba_obs_forward = np.sum(self.alpha[-1])
            proba_obs_backward = np.sum(self.init_distr*self.beta[0]*self.B[:, 0])
        assert np.allclose(proba_obs_forward, proba_obs_backward), "Find forward: {}, backward: {}".format(proba_obs_forward, proba_obs_backward)
        return True
    def compute_gamma_zeta(self, Obs):
        """
        No "for loop" implementation for conditional proba !
        gamma: P(i_t = q_i| O, lambda), shape = (T, num_s)
        zeta : P(i_t = q_i, i_t+1 = q_j| O, lambda), shape = (T, num_s, num_s)

        """
        # import ipdb; ipdb.set_trace()
        if self.use_log:
            
            gamma = self.alpha + self.beta ## shape = (T, num_s) 
            # max_state = np.max(gamma, axis=1, keepdims=True) ## shape = (T, 1)
            # denominator = max_state + np.log(np.sum(np.exp(gamma - max_state), axis=1, keepdims=True)) ## shape = (T,1)
            denominator = self.compute_log_sum(gamma, axis=1)
            gamma = gamma - denominator

            part1 = self.alpha[..., None] + self.A
            part2 = self.B.T + self.beta ## shape = (T, num_s)
            part1 = part1[:-1] # shape = (T-1, num_s, num_s)
            part2 = part2[1:] # shape = (T-1, num_s)

            zeta = part1 + part2[:, None, :]## shape = (T-1, num_s, num_s)
            # max_states = np.max(zeta, axis=(-2,-1), keepdims=True) ## shape (T-1, 1, 1)
            # denominator = max_states + np.log(np.sum(np.exp(zeta - max_states), axis=(1,2), keepdims=True))
            denominator = self.compute_log_sum(zeta, axis=(1,2))
            zeta = zeta - denominator ## shape = (T-1, num_s, num_s)

        else:
            gamma = self.alpha*self.beta ## shape = (T, num_s)
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True) ## shape = (T, num_s)

            part1 = self.alpha[..., None]*self.A ## shape = (T, num_s, num_s), since (T, N, 1) * (N, N) = (T, N, N)
            part2 = self.B.T*self.beta ## shape = (T, num_s)
            
            part1 = part1[:-1] # shape = (T-1, num_s, num_s)
            part2 = part2[1:] # shape = (T-1, num_s)

            zeta = part1*part2[:, None, :]## shape = (T-1, num_s, num_s)
            zeta = zeta/np.sum(zeta, axis=(1,2), keepdims=True) ## shape = (T-1, num_s, num_s)
        # import ipdb; ipdb.set_trace()
        self.gamma = gamma
        self.zeta = zeta

    def compute_log_sum(self, X, axis=0):
        """
        Suppose that X is already in log-scale

        keepdims is always true !
        """
        max_tempo = np.max(X, axis=axis, keepdims=True)
        return max_tempo + np.log(np.sum(np.exp(X-max_tempo), axis=axis, keepdims=True))

    def update_params(self, Obs):
        """
        Update parameters using forward/backward estimation: init_distr, A, B

        """


        if self.use_log:
            new_init_distr = self.gamma[0]
            # max_T_zeta = np.max(self.zeta, axis=0, keepdims=True) ## shape (1, num_s, num_s)
            # part1 = np.squeeze(max_T_zeta)+np.log(np.sum(np.exp(self.zeta - max_T_zeta), axis=0)) # shape=(num_s, num_s)
            part1 = np.squeeze(self.compute_log_sum(self.zeta, axis=0)) ## shape = (num_s, num_s)
            part2 = np.squeeze(self.compute_log_sum(self.gamma[:-1], axis=0))[:, None] ## shape (num_s, 1)
            new_A = part1 - part2

            new_B_mu = np.sum(np.exp(self.gamma)[...,None]*np.expand_dims(Obs, axis=1), axis=0) ## shape = (num_s, dim)
            new_B_mu = new_B_mu / (np.sum(np.exp(self.gamma), axis=0)[:, None])
            
            centered_part = Obs - np.expand_dims(new_B_mu, axis=1) ## shape (T, dim) - (num_s, 1, dim) = (num_s, T, dim)
            sigma_part = centered_part[..., None] * np.expand_dims(centered_part, axis=2) ## shape = (num_s, T, dim , dim)

            new_B_sigma = np.sum(sigma_part* (np.exp(self.gamma.T))[..., None, None], axis=1) ## shape = (num_s, dim, dim)
            new_B_sigma = new_B_sigma/ (np.sum(np.exp(self.gamma), axis=0)[:, None, None])

        else:
            new_init_distr = self.gamma[0]
            new_A = np.sum(self.zeta, axis=0)/(np.sum(self.gamma[:-1], axis=0)[:, None])

            new_B_mu = np.sum(self.gamma[...,None]*np.expand_dims(Obs, axis=1), axis=0) ## shape = (num_s, dim)
            new_B_mu = new_B_mu / (np.sum(self.gamma, axis=0)[:, None])
            
            centered_part = Obs - np.expand_dims(new_B_mu, axis=1) ## shape (T, dim) - (num_s, 1, dim) = (num_s, T, dim)
            sigma_part = centered_part[..., None] * np.expand_dims(centered_part, axis=2) ## shape = (num_s, T, dim , dim)

            new_B_sigma = np.sum(sigma_part* (self.gamma.T)[..., None, None], axis=1) ## shape = (num_s, dim, dim)
            new_B_sigma = new_B_sigma/ (np.sum(self.gamma, axis=0)[:, None, None])


        self.init_distr = new_init_distr
        self.A = new_A
        self.B_mu = new_B_mu ## shape = (num_s, dim)
        self.B_sigma = new_B_sigma ## shape = (num_s, dim, dim)
    def multivariate_gaussian_pdf_simple(self, X, mu, sigma):
        """
        X.shape = (N, dim)
        mu.shape = (dim,)
        sigma.shape = (dim, dim)

        return shape = (N,)

        """
        dim = X.shape[-1]
        return 1/np.sqrt((2*np.pi)**dim * np.linalg.det(sigma)) * np.exp(-0.5*np.sum((X-mu).dot(np.linalg.inv(sigma))*(X-mu), axis=1))

    def compute_log_likelihood_GMM(self, X, pi):    
        cum = 0
        tempo_dist = np.empty((len(X), self.num_s))
        for i in range(self.num_s):
            tempo_dist[:, i] = np.sum(np.square(X-self.B_mu[i]), axis=1)
        
        ### reset self.clusters_attribution for the function "plot_datasets"
        
        ######### GMM (Gaussian mixture model) estimation of states distribution
        # self.clusters_attribution = np.argmin(tempo_dist, axis=1) ## shape=(len(newX),)
        # frequency = (self.clusters_attribution == np.arange(self.num_s)[:, None]) ## shape = (num_s, T)
        # pi_gauss = np.mean(frequency, axis=1)
        # import ipdb; ipdb.set_trace()

        ### E_step
        Q_t = np.zeros((len(X), self.num_s))
        for k in range(self.num_s):
            Q_t[:, k] = pi[k]*self.multivariate_gaussian_pdf_simple(X, self.B_mu[k], self.B_sigma[k])

        Q_t = Q_t/np.sum(Q_t, axis=1, keepdims=True)
        H_q = -np.sum(Q_t*np.log(Q_t))## - q log(q)
        cum = np.sum(Q_t*np.log(pi)) + H_q 

        for k in range(self.num_s):
            A = np.log(self.multivariate_gaussian_pdf_simple(X, self.B_mu[k], self.B_sigma[k]))
            cum += np.sum(Q_t[:, k]*A)
        return cum

    def compute_log_likelihood_HMM(self, newX=None):
        
        
        part1 = np.sum(self.compute_log_sum(self.gamma[0] + self.init_distr))
        part2 = np.sum(self.compute_log_sum(self.zeta + self.A, axis=(1,2)))
        if newX is None:
            part3 = np.sum(self.compute_log_sum(self.gamma + self.B.T, axis=1))
        else:
            cum = 0
            B = self.multivariate_gaussian_pdf(newX, self.B_mu, self.B_sigma, log_scale=True) ## shape = (num_s, T)
            info = viterbi_hmm_continue(self.init_distr, self.A, B, newX, use_log=True)

            newB = np.zeros(len(newX))
            for i in range(len(newX)):
                k = int(info[0][i])                
                newB[i] = np.log(self.multivariate_gaussian_pdf_simple(newX[i][None, :], self.B_mu[k], self.B_sigma[k]))

            part3 = np.sum(self.compute_log_sum(self.gamma + newB[:, None], axis=1))
        # import ipdb; ipdb.set_trace()
        return part1 + part2 + part3

    def main(self, iterations, Obs):
        
        for i in range(iterations):
            self.build_emission_proba(Obs, log_scale=self.use_log) ## build B based on (B_mu, B_sigma) and Observations
            self.forward(Obs) ## use (init_distr, A, B) to estimate alpha 
            self.backward(Obs) ## use (init_distr, A, B) to estimate beta 
            self.compute_gamma_zeta(Obs) ### use (alpha, beta) to estimate (gamma, zeta)
            self.update_params(Obs) ## update (init_distr, A, B_mu, B_sigma)
if __name__=="__main__":
	print("Start")
