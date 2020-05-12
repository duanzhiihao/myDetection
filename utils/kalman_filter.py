# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


class KalmanFilter(object):
    def __init__(self, nparam, initial_P_cov, Q_cov, R_cov,
                       auto_noise:list=None):
        assert len(initial_P_cov) == len(Q_cov) == 2*nparam and len(R_cov) == nparam
        # F: state-transition model matrix
        self._F = np.eye(2 * nparam, 2 * nparam)
        for i in range(nparam):
            self._F[i, nparam + i] = 1
        # H: observation matrix
        self._H = np.eye(nparam, 2 * nparam)
        # initial covariance matrix
        self._P0 = np.diag(np.square(initial_P_cov))
        # prediction uncertainty
        self._Q = np.diag(np.square(Q_cov))
        # measurement uncertainty
        self._R = np.diag(np.square(R_cov))

        if auto_noise is not None:
            assert len(auto_noise) == nparam
            assert all([isinstance(b, bool) for b in auto_noise])
            self.auto_noise = np.array(auto_noise*2, dtype=np.bool)
        else:
            self.auto_noise = None
        self.nparam = nparam

    def initiate(self, measurement):
        x = np.r_[measurement, np.zeros_like(measurement)]
        P = self._P0.copy()
        if self.auto_noise is not None:
            P[self.auto_noise] *= np.sqrt(x[2] * x[3])
        
        self.x = x.copy()
        self.P = P.copy()

    def predict(self):
        x, P = self.x, self.P
        Q = self._Q.copy()
        if self.auto_noise is not None:
            Q[self.auto_noise] *= np.sqrt(x[2] * x[3])
        
        x = self._F @ x + 0
        P = np.linalg.multi_dot([self._F, P, self._F.T]) + Q

        self.x = x.copy()
        self.P = P.copy()
        return x[:self.nparam]

    def update(self, measurement):
        x, P = self.x, self.P
        R = self._R.copy()
        if self.auto_noise is not None:
            R[self.auto_noise[:self.nparam]] *= np.sqrt(x[2] * x[3])
        
        y = measurement - self._H @ x
        S = np.linalg.multi_dot((self._H, P, self._H.T)) + R

        # chol_factor, lower = scipy.linalg.cho_factor(
        #     S, lower=True, check_finite=False)
        # K = scipy.linalg.cho_solve((chol_factor, lower), np.dot(P, self._H.T).T,
        #                            check_finite=False).T
        K = np.linalg.multi_dot((P, self._H.T, np.linalg.inv(S)))

        est_mean = x + K @ y
        # est_cov = P - np.linalg.multi_dot((K, S, K.T))
        est_cov = P - np.linalg.multi_dot((K, self._H, P))

        self.x = est_mean.copy()
        self.P = est_cov.copy()
        return est_mean[:self.nparam]