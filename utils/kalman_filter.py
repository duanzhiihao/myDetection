# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


class KalmanFilter(object):
    def __init__(self, nparam, initial_P_cov, Q_cov, R_cov):
        assert nparam == 4
        assert len(initial_P_cov) == len(Q_cov) == 2*nparam and len(R_cov) == nparam
        # F: state-transition model matrix
        self._F = np.eye(2 * nparam, 2 * nparam)
        for i in range(nparam):
            self._F[i, nparam + i] = 1
        # H: observation matrix
        self._H = np.eye(nparam, 2 * nparam)
        # initial covariance matrix
        self._inital_P = np.diag(np.square(initial_P_cov))
        # prediction uncertainty
        self._Q = np.diag(np.square(Q_cov))
        # measurement uncertainty
        self._R = np.diag(np.square(R_cov))

        self.nparam = nparam

    def initiate(self, measurement):
        x = np.r_[measurement, np.zeros_like(measurement)]
        P = self._inital_P * np.sqrt(x[2] * x[3])
        self.x = x.copy()
        self.P = P.copy()

    def predict(self):
        x, P = self.x, self.P
        Q = self._Q * np.sqrt(x[2] * x[3])
        #x = np.dot(self._F, x)
        x = self._F @ x + 0
        P = np.linalg.multi_dot([self._F, P, self._F.T]) + Q
        self.x = x.copy()
        self.P = P.copy()
        return x[:self.nparam]

    def update(self, measurement):
        x, P = self.x, self.P
        
        y = measurement - self._H @ x
        R = self._R * np.sqrt(x[2] * x[3])
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
