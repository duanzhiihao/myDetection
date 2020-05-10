# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


class XYWHKalmanFilter(object):
    def __init__(self, nparam):
        assert nparam == 4
        # F: state-transition model matrix
        self._F = np.eye(2 * nparam, 2 * nparam)
        for i in range(nparam):
            self._F[i, nparam + i] = 1
        # H: observation matrix
        self._H = np.eye(nparam, 2 * nparam)

        # std_pos = 1 / 20
        # std_v = 1 / 160
        # initial covariance matrix
        diagonal = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self._inital_P = np.diag(np.square(diagonal))
        # prediction uncertainty
        diagonal = [0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01]
        self._Q = np.diag(np.square(diagonal))
        # measurement uncertainty
        diagonal = [0.01, 0.01, 0.05, 0.05]
        self._R = np.diag(np.square(diagonal))

        self.nparam = nparam

    def initiate(self, measurement):
        x = np.r_[measurement, np.zeros_like(measurement)]
        P = self._inital_P * np.sqrt(x[2] * x[3])
        self.x = x.copy()
        self.P = P.copy()

    def predict(self):
        x, P = self.x, self.P
        # std_pos = [self._std_weight_position * x[3]] * self.nparam
        # std_vel = [self._std_weight_velocity * x[3]] * self.nparam
        # Q = np.diag(np.square(np.r_[std_pos, std_vel]))
        Q = self._Q * np.sqrt(x[2] * x[3])
        #x = np.dot(self._F, x)
        x = x @ self._F.T
        P = np.linalg.multi_dot([self._F, P, self._F.T]) + Q
        self.x = x.copy()
        self.P = P.copy()
        return x[:self.nparam]

    def update(self, measurement):
        x, P = self.x, self.P
        
        y = measurement - self._H @ x
        # std = [self._std_weight_position * x[3]] * self.nparam
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


# import scipy.linalg
# class KFTracklet_():
#     '''
#     Tracklet with Kalman Filter (KF).

#     The notation of the KF parameters are consistent with the wikipedia page: \
#     https://en.wikipedia.org/wiki/Kalman_filter

#     Args:
#         bbox: initial bounding box
#     '''
#     def __init__(self, bbox, object_id, global_step=0, img_hw=None):
#         assert isinstance(bbox, np.ndarray) and bbox.ndim == 1
#         nparam = bbox.shape[0]

#         # F: state-transition model matrix
#         self._F = np.eye(2*nparam, 2*nparam)
#         for i in range(nparam):
#             self._F[i, nparam + i] = 1
#         # H: observation matrix
#         self._H = np.eye(nparam, 2 * nparam)
#         # uncertainty weights
#         self._std_weight_position = 1 / 20
#         self._std_weight_velocity = 1 / 160

#         #x: state vector
#         self.x = np.concatenate([bbox, np.zeros_like(bbox)]).reshape(2*nparam,1)
#         _std = [
#             2 * self._std_weight_position * bbox[3],
#             2 * self._std_weight_position * bbox[3],
#             2 * self._std_weight_position * bbox[3],
#             2 * self._std_weight_position * bbox[3],
#             2 * self._std_weight_position * bbox[3],
#             10 * self._std_weight_velocity * bbox[3],
#             10 * self._std_weight_velocity * bbox[3],
#             10 * self._std_weight_velocity * bbox[3],
#             10 * self._std_weight_velocity * bbox[3],
#             10 * self._std_weight_velocity * bbox[3]]
#         # P matrix
#         self.P = np.diag(np.square(_std))

#         self.object_id = object_id
#         self.step = global_step
#         self.img_hw = img_hw
#         self._pred_flag = False
    
#     def _docs(self):
#         raise Exception()
#         # self._F = np.array([[1,0,0,0,1,0,0,0],
#         #                    [0,1,0,0,0,1,0,0],
#         #                    [0,0,1,0,0,0,1,0],
#         #                    [0,0,0,1,0,0,0,1],
#         #                    [0,0,0,0,1,0,0,0],
#         #                    [0,0,0,0,0,1,0,0],
#         #                    [0,0,0,0,0,0,1,0],
#         #                    [0,0,0,0,0,0,0,1]])
#         # self._H = np.array([[1,0,0,0,0,0,0,0],
#         #                    [0,1,0,0,0,0,0,0],
#         #                    [0,0,1,0,0,0,0,0],
#         #                    [0,0,0,1,0,0,0,0],])


#     def predict(self):
#         x, P = self.x, self.P
#         std_pos = [
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3]]
#         std_vel = [
#             self._std_weight_velocity * x[3],
#             self._std_weight_velocity * x[3],
#             self._std_weight_velocity * x[3],
#             self._std_weight_velocity * x[3],
#             self._std_weight_velocity * x[3]]
        
#         # P of the process noise
#         Q = np.diag(np.square(np.r_[std_pos, std_vel]))
#         x = self._F @ x + 0
#         P = np.linalg.multi_dot([self._F, P, self._F.T]) + Q

#         self.x = x
#         self.P = P
#         self.step += 1
#         self._pred_flag = True
#         return x[:5].squeeze(1)
    
#     def update(self, z) -> np.ndarray:
#         assert isinstance(z, np.ndarray) and z.ndim == 1
#         assert self._pred_flag, 'Please call predict() before update()'
#         z = z.reshape(-1,1)
#         x, P = self.x, self.P
#         _std = [
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3],
#             self._std_weight_position * x[3]]
#         # R: P of obervation noise
#         R = np.diag(np.square(_std))

#         Hx = self._H @ x
#         S = np.linalg.multi_dot([self._H, P, self._H.T]) + R

#         chol_factor, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
#         K = scipy.linalg.cho_solve((chol_factor, lower), (P @ self._H.T).T,
#                                    check_finite=False).T
        
#         x = x + K @ (z - Hx)
#         assert np.isclose(np.linalg.multi_dot((K, S, K.T)),
#                           np.linalg.multi_dot((K, self._H, P))).all()
#         P = P - np.linalg.multi_dot((K, S, K.T))
#         # P = P - np.linalg.multi_dot((K, self._H, P))
        
#         self.x = x
#         self.P = P
#         self._pred_flag = False
#         return x[:5].squeeze(1)

#     def is_feasible(self):
#         imh, imw = self.img_hw
#         bbox = self.x[:5]
#         if (bbox[:4] < 0).any():
#             return False
#         if (bbox[0] > imw) or (bbox[1] > imh) or (bbox[2] > imw) or (bbox[3] > imh):
#             return False
#         return True