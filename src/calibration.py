import numpy as np
import pandas as pd
import json
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
from scipy.optimize import least_squares
from pathlib import Path

# Units used throughout the calibration pipeline.
ROBOT_UNIT = "mm"   # Robot x, y, z unit.
NDI_UNIT   = "mm"   # NDI tx, ty, tz unit.


class HandEyeCalibration:
    """Eye-to-hand calibration using NDI marker observations and robot poses.

    Frames:
        - NDI camera frame
        - Robot base frame
        - Robot end-effector frame
        - Marker frame attached to the end effector

    Forward model:
        T_ndi_marker = T_ndi_base @ T_base_ee @ T_ee_marker

    Navigation formula:
        T_base_ee_target = inv(T_ndi_base) @ T_ndi_marker_new @ inv(T_ee_marker)
    """

    def __init__(self, csv_path='../dataset/calibration/calibration_data.csv'):
        self.csv_path = csv_path
        self.raw_data = None
        self.averaged_data = None
        self.all_data = None
        self.pose_cache = None
        self.T_ndi_base = None
        self.T_ee_marker = None
        self.method_name = None
        self.ndi_position_bias = np.zeros(3)
        self.ndi_axis_scale = np.ones(3)

    # 1. Data loading and preprocessing
    def load_and_preprocess_data(self):
        print("=" * 60)
        print("1) Data Loading and Preprocessing")
        print("=" * 60)
        print(f"  Robot unit: {ROBOT_UNIT}, NDI unit: {NDI_UNIT}")

        self.raw_data = pd.read_csv(self.csv_path)
        print(f"  Raw rows: {len(self.raw_data)}")

        grouped = self.raw_data.groupby('pose_id')
        averaged_list = []
        for pose_id, group in grouped:
            ndi_quats = group[['qx', 'qy', 'qz', 'q0']].to_numpy(dtype=float)
            avg_quat = self._average_quaternions(ndi_quats)
            robot_quats = R.from_euler(
                'ZYX',
                group[['w', 'v', 'u']].to_numpy(dtype=float),
                degrees=True).as_quat()
            avg_robot_quat = self._average_quaternions(robot_quats)
            avg_robot_zyx = R.from_quat(avg_robot_quat).as_euler('ZYX', degrees=True)
            avg_data = {
                'pose_id': pose_id,
                'q0': avg_quat[3], 'qx': avg_quat[0],
                'qy': avg_quat[1], 'qz': avg_quat[2],
                'tx': group['tx'].median(),
                'ty': group['ty'].median(),
                'tz': group['tz'].median(),
                'x':  group['x'].median(),
                'y':  group['y'].median(),
                'z':  group['z'].median(),
                'u':  avg_robot_zyx[2],
                'v':  avg_robot_zyx[1],
                'w':  avg_robot_zyx[0],
            }
            averaged_list.append(avg_data)

        self.averaged_data = pd.DataFrame(averaged_list).reset_index(drop=True)
        print(f"  Averaged poses: {len(self.averaged_data)}")

        if 'error' in self.raw_data.columns:
            err = self.raw_data['error']
            print(f"  NDI measurement error: mean={err.mean():.4f}, "
                  f"max={err.max():.4f}, std={err.std():.4f}")

        print("\n  [Data Range Check]")
        print(f"  Robot xyz range: "
              f"x=[{self.averaged_data['x'].min():.1f}, {self.averaged_data['x'].max():.1f}], "
              f"y=[{self.averaged_data['y'].min():.1f}, {self.averaged_data['y'].max():.1f}], "
              f"z=[{self.averaged_data['z'].min():.1f}, {self.averaged_data['z'].max():.1f}]")
        print(f"  NDI txyz range: "
              f"tx=[{self.averaged_data['tx'].min():.1f}, {self.averaged_data['tx'].max():.1f}], "
              f"ty=[{self.averaged_data['ty'].min():.1f}, {self.averaged_data['ty'].max():.1f}], "
              f"tz=[{self.averaged_data['tz'].min():.1f}, {self.averaged_data['tz'].max():.1f}]")
        print()

        self.all_data = self.averaged_data.copy()
        self.pose_cache = self._build_pose_cache(self.all_data)
        print(f"  Working poses: {len(self.all_data)}\n")

    def _average_quaternions(self, quats_xyzw: np.ndarray) -> np.ndarray:
        """Average quaternions with the Markley eigendecomposition method."""
        M = np.zeros((4, 4))
        for q in quats_xyzw:
            q = q / np.linalg.norm(q)
            q_wxyz = np.array([q[3], q[0], q[1], q[2]])
            M += np.outer(q_wxyz, q_wxyz)
        M /= len(quats_xyzw)
        eigvals, eigvecs = np.linalg.eigh(M)
        avg_wxyz = eigvecs[:, np.argmax(eigvals)]
        return np.array([avg_wxyz[1], avg_wxyz[2], avg_wxyz[3], avg_wxyz[0]])

    # 2. Transform helpers and pose cache
    def _make_transform_batch(self, rotations, translations):
        """Build a stack of homogeneous transforms from batched rotations/translations."""
        n = len(translations)
        transforms = np.broadcast_to(np.eye(4, dtype=float), (n, 4, 4)).copy()
        transforms[:, :3, :3] = rotations
        transforms[:, :3, 3] = translations
        return transforms

    def _invert_transform_batch(self, transforms):
        """Fast inverse for a stack of rigid transforms."""
        rotations_t = np.transpose(transforms[:, :3, :3], (0, 2, 1))
        translations = -np.einsum('nij,nj->ni', rotations_t, transforms[:, :3, 3])
        inv_transforms = np.broadcast_to(np.eye(4, dtype=float), transforms.shape).copy()
        inv_transforms[:, :3, :3] = rotations_t
        inv_transforms[:, :3, 3] = translations
        return inv_transforms

    def _build_pose_cache(self, data):
        """Precompute per-pose transforms so calibration does not rebuild them repeatedly."""
        if data is None or len(data) == 0:
            empty_R = np.empty((0, 3, 3), dtype=float)
            empty_t = np.empty((0, 3), dtype=float)
            empty_T = np.empty((0, 4, 4), dtype=float)
            return {
                'count': 0,
                'ndi_R': empty_R,
                'ndi_t': empty_t,
                'robot_R': empty_R,
                'robot_t': empty_t,
                'ndi_T': empty_T,
                'robot_T': empty_T,
                'ndi_T_inv': empty_T,
                'robot_T_inv': empty_T,
            }

        quat_xyzw = data[['qx', 'qy', 'qz', 'q0']].to_numpy(dtype=float)
        ndi_t = data[['tx', 'ty', 'tz']].to_numpy(dtype=float)
        robot_t = data[['x', 'y', 'z']].to_numpy(dtype=float)
        robot_euler_zyx = data[['w', 'v', 'u']].to_numpy(dtype=float)

        ndi_R = R.from_quat(quat_xyzw).as_matrix()
        robot_R = R.from_euler('ZYX', robot_euler_zyx, degrees=True).as_matrix()
        ndi_T = self._make_transform_batch(ndi_R, ndi_t)
        robot_T = self._make_transform_batch(robot_R, robot_t)

        return {
            'count': len(data),
            'ndi_R': ndi_R,
            'ndi_t': ndi_t,
            'robot_R': robot_R,
            'robot_t': robot_t,
            'ndi_T': ndi_T,
            'robot_T': robot_T,
            'ndi_T_inv': self._invert_transform_batch(ndi_T),
            'robot_T_inv': self._invert_transform_batch(robot_T),
        }

    def _get_pose_cache(self, data):
        """Return cached transforms for the dataset."""
        if data is self.all_data or data is self.averaged_data:
            cache = self.pose_cache
        else:
            cache = self._build_pose_cache(data)
        return cache

    def _average_transform_batch(self, transforms):
        """Average SE(3) poses with a Lie-algebra mean over the rotation part."""
        if len(transforms) == 0:
            return np.eye(4)
        if len(transforms) == 1:
            return transforms[0].copy()

        R_ref = transforms[0, :3, :3]
        dR_logs = []
        for transform in transforms:
            dR = R_ref.T @ transform[:3, :3]
            dR_logs.append(np.real(logm(dR)))

        R_mean = R_ref @ expm(np.mean(np.asarray(dR_logs), axis=0))
        t_mean = np.mean(transforms[:, :3, 3], axis=0)
        return self.create_homogeneous_matrix(R_mean, t_mean)

    def _predict_robot_pose_batch(self, cache, T_ndi_base, T_ee_marker,
                                  ndi_position_bias=None, ndi_axis_scale=None):
        """Predict all robot poses from cached NDI measurements in one batched pass."""
        R_bn = T_ndi_base[:3, :3].T
        t_bn = -R_bn @ T_ndi_base[:3, 3]
        R_me = T_ee_marker[:3, :3].T
        t_me = -R_me @ T_ee_marker[:3, 3]

        ndi_t = cache['ndi_t'].copy()
        if ndi_axis_scale is not None:
            ndi_t *= np.asarray(ndi_axis_scale, dtype=float).reshape(1, 3)
        if ndi_position_bias is not None:
            ndi_t += np.asarray(ndi_position_bias, dtype=float).reshape(1, 3)

        R_pred = np.einsum('ab,nbc,cd->nad', R_bn, cache['ndi_R'], R_me)
        marker_offset = np.einsum('nij,j->ni', cache['ndi_R'], t_me)
        t_pred = np.einsum('ab,nb->na', R_bn, marker_offset + ndi_t) + t_bn
        return R_pred, t_pred

    def quaternion_to_rotation_matrix(self, q0, qx, qy, qz):
        return R.from_quat([qx, qy, qz, q0]).as_matrix()

    def euler_to_rotation_matrix(self, u_deg, v_deg, w_deg):
        """Neuromeka INDY: u=Rx, v=Ry, w=Rz, extrinsic XYZ = intrinsic ZYX"""
        return R.from_euler('ZYX', [w_deg, v_deg, u_deg], degrees=True).as_matrix()

    def create_homogeneous_matrix(self, R_mat, t):
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = np.asarray(t).flatten()
        return T

    def get_ndi_transform(self, row):
        """Build T_ndi_marker from one NDI measurement row."""
        R_mat = self.quaternion_to_rotation_matrix(
            row['q0'], row['qx'], row['qy'], row['qz'])
        t = np.array([row['tx'], row['ty'], row['tz']])
        return self.create_homogeneous_matrix(R_mat, t)

    def get_ndi_transform_with_bias(self, row, ndi_position_bias=None,
                                    ndi_axis_scale=None):
        """Build T_ndi_marker after applying NDI scale and bias correction.

        t_corr = diag(scale) @ t_raw + bias
        """
        T_ndi = self.get_ndi_transform(row)
        t = T_ndi[:3, 3].copy()
        if ndi_axis_scale is not None:
            t = np.asarray(ndi_axis_scale).reshape(3) * t
        if ndi_position_bias is not None:
            t = t + np.asarray(ndi_position_bias).reshape(3)
        T_ndi[:3, 3] = t
        return T_ndi

    def get_robot_transform(self, row):
        """Build T_base_ee from one robot pose row."""
        R_mat = self.euler_to_rotation_matrix(row['u'], row['v'], row['w'])
        t = np.array([row['x'], row['y'], row['z']])
        return self.create_homogeneous_matrix(R_mat, t)

    # 3. Hand-eye initialization
    def solve_handeye_for_T_ee_marker(self, data):
        """Estimate T_ee_marker with OpenCV PARK hand-eye calibration."""
        cache = self._get_pose_cache(data)
        R_g2b = [rot for rot in cache['robot_R']]
        t_g2b = [vec.reshape(3, 1) for vec in cache['robot_t']]
        R_t2c = [rot for rot in cache['ndi_T_inv'][:, :3, :3]]
        t_t2c = [vec.reshape(3, 1) for vec in cache['ndi_T_inv'][:, :3, 3]]
        R_out, t_out = cv2.calibrateHandEye(
            R_gripper2base=R_g2b, t_gripper2base=t_g2b,
            R_target2cam=R_t2c, t_target2cam=t_t2c,
            method=cv2.CALIB_HAND_EYE_PARK)
        return self.create_homogeneous_matrix(R_out, t_out.flatten())

    # 4. Closed-form base transform estimation
    def compute_T_ndi_base_from_T_ee_marker(self, data, T_ee_marker):
        """Estimate T_ndi_base by averaging per-pose closed-form solutions."""
        cache = self._get_pose_cache(data)
        T_ee_marker_inv = np.linalg.inv(T_ee_marker)
        T_est = np.matmul(np.matmul(cache['ndi_T'], T_ee_marker_inv), cache['robot_T_inv'])
        return self._average_transform_batch(T_est)

    # 5. Point registration refinement
    def solve_point_registration(self, data, T_ee_marker):
        """Estimate T_ndi_base with SVD point registration on marker origins."""
        cache = self._get_pose_cache(data)
        p_ndi = cache['ndi_t']
        p_base = np.einsum('nij,j->ni', cache['robot_R'], T_ee_marker[:3, 3]) + cache['robot_t']

        centroid_ndi = p_ndi.mean(axis=0)
        centroid_base = p_base.mean(axis=0)
        H = (p_base - centroid_base).T @ (p_ndi - centroid_ndi)
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R_reg = Vt.T @ np.diag([1, 1, d]) @ U.T
        t_reg = centroid_ndi - R_reg @ centroid_base

        T_ndi_base = self.create_homogeneous_matrix(R_reg, t_reg)
        p_pred = (R_reg @ p_base.T).T + t_reg
        residuals = np.linalg.norm(p_pred - p_ndi, axis=1)
        return T_ndi_base, residuals

    # 6. Joint nonlinear refinement
    def _rotvec_t_to_matrix(self, rotvec, t):
        """Convert a rotation vector and translation into a 4x4 transform."""
        R_mat = R.from_rotvec(rotvec).as_matrix()
        return self.create_homogeneous_matrix(R_mat, t)

    def refine_nonlinear_with_ndi_axis_scale(self, data, T_ndi_base_init, T_ee_marker_init,
                                             ndi_position_bias_init=None,
                                             ndi_axis_scale_init=None,
                                             pos_weight=1.0, rot_weight=20.0,
                                             bias_reg_weight=0.02,
                                             scale_reg_weight=10.0):
        """Refine both transforms together with NDI scale and bias parameters.

        The scale term is regularized toward 1.0 to avoid unrealistic stretching.
        """
        cache = self._get_pose_cache(data)
        ndi_R = cache['ndi_R']
        robot_R = cache['robot_R']
        robot_R_t = np.transpose(robot_R, (0, 2, 1))
        ndi_t_raw = cache['ndi_t']
        robot_t = cache['robot_t']

        rv_ndi = R.from_matrix(T_ndi_base_init[:3, :3]).as_rotvec()
        t_ndi = T_ndi_base_init[:3, 3]
        rv_ee = R.from_matrix(T_ee_marker_init[:3, :3]).as_rotvec()
        t_ee = T_ee_marker_init[:3, 3]
        b0 = np.zeros(3) if ndi_position_bias_init is None else np.asarray(ndi_position_bias_init).reshape(3)
        s0 = np.ones(3) if ndi_axis_scale_init is None else np.asarray(ndi_axis_scale_init).reshape(3)
        x0 = np.concatenate([rv_ndi, t_ndi, rv_ee, t_ee, b0, s0])

        def residual_fn(x):
            R_nb = R.from_rotvec(x[0:3]).as_matrix()
            t_nb = x[3:6]
            R_em = R.from_rotvec(x[6:9]).as_matrix()
            t_em = x[9:12]
            b_ndi = x[12:15]
            s_ndi = x[15:18]

            R_bn = R_nb.T
            t_bn = -R_bn @ t_nb
            R_me = R_em.T
            t_me = -R_me @ t_em

            ndi_t = ndi_t_raw * s_ndi.reshape(1, 3) + b_ndi.reshape(1, 3)
            R_pred = np.einsum('ab,nbc,cd->nad', R_bn, ndi_R, R_me)
            marker_offset = np.einsum('nij,j->ni', ndi_R, t_me)
            t_pred = np.einsum('ab,nb->na', R_bn, marker_offset + ndi_t) + t_bn

            pos_err = (t_pred - robot_t) * pos_weight
            R_err = np.einsum('nij,njk->nik', R_pred, robot_R_t)
            rv_err = R.from_matrix(R_err).as_rotvec() * rot_weight
            regularizer = np.concatenate([
                bias_reg_weight * b_ndi,
                scale_reg_weight * (s_ndi - np.ones(3)),
            ])
            return np.concatenate([np.hstack((pos_err, rv_err)).ravel(),
                                   regularizer])

        result = least_squares(residual_fn, x0, method='lm', max_nfev=40000)
        T_ndi_base_opt = self._rotvec_t_to_matrix(result.x[0:3], result.x[3:6])
        T_ee_marker_opt = self._rotvec_t_to_matrix(result.x[6:9], result.x[9:12])
        ndi_bias_opt = result.x[12:15]
        ndi_scale_opt = result.x[15:18]
        return T_ndi_base_opt, T_ee_marker_opt, ndi_bias_opt, ndi_scale_opt, result.cost

    # 7. Error evaluation
    def evaluate_absolute_position(self, data, T_ndi_base, T_ee_marker,
                                   ndi_position_bias=None, ndi_axis_scale=None):
        """Measure absolute robot-pose error induced by the calibration result.

        Predicted robot pose:
            T_base_ee_pred = inv(T_ndi_base) @ T_ndi_marker @ inv(T_ee_marker)
        """
        cache = self._get_pose_cache(data)
        R_pred, t_pred = self._predict_robot_pose_batch(
            cache, T_ndi_base, T_ee_marker,
            ndi_position_bias=ndi_position_bias,
            ndi_axis_scale=ndi_axis_scale)

        pos_errors = np.linalg.norm(t_pred - cache['robot_t'], axis=1)
        R_err = np.einsum('nij,njk->nik', np.transpose(R_pred, (0, 2, 1)), cache['robot_R'])
        angles = np.arccos(np.clip((np.trace(R_err, axis1=1, axis2=2) - 1) / 2, -1, 1))
        return pos_errors, np.degrees(angles)

    # 8. Final result handling
    def _set_final_result(self, method_name, T_ndi_base, T_ee_marker,
                          pos_errors, rot_errors,
                          ndi_position_bias=None, ndi_axis_scale=None):
        self.method_name = method_name
        self.T_ndi_base = T_ndi_base
        self.T_ee_marker = T_ee_marker
        self.ndi_position_bias = np.zeros(3) if ndi_position_bias is None else np.asarray(ndi_position_bias).copy()
        self.ndi_axis_scale = np.ones(3) if ndi_axis_scale is None else np.asarray(ndi_axis_scale).copy()

        pos_mean = float(np.mean(pos_errors))
        pos_max = float(np.max(pos_errors))
        pos_std = float(np.std(pos_errors))
        rot_mean = float(np.mean(rot_errors))

        print("=" * 90)
        print(f"Final Fast Calibration Result ({self.method_name})")
        print("=" * 90)
        print(f"  Position error: mean={pos_mean:.4f}, max={pos_max:.4f}, std={pos_std:.4f}{ROBOT_UNIT}")
        print(f"  Rotation error: mean={rot_mean:.4f} deg\n")

        euler_ndi = R.from_matrix(self.T_ndi_base[:3, :3]).as_euler('ZYX', degrees=True)
        euler_ee = R.from_matrix(self.T_ee_marker[:3, :3]).as_euler('ZYX', degrees=True)
        print("T_ndi_base (NDI -> Robot base):")
        print(f"  t=[{self.T_ndi_base[0,3]:.4f}, {self.T_ndi_base[1,3]:.4f}, "
              f"{self.T_ndi_base[2,3]:.4f}]{ROBOT_UNIT}")
        print(f"  R(ZYX)=[{euler_ndi[0]:.4f} deg, {euler_ndi[1]:.4f} deg, {euler_ndi[2]:.4f} deg]")
        print("T_ee_marker (EE -> Marker):")
        print(f"  t=[{self.T_ee_marker[0,3]:.4f}, {self.T_ee_marker[1,3]:.4f}, "
              f"{self.T_ee_marker[2,3]:.4f}]{ROBOT_UNIT}")
        print(f"  R(ZYX)=[{euler_ee[0]:.4f} deg, {euler_ee[1]:.4f} deg, {euler_ee[2]:.4f} deg]")
        print(f"NDI position bias: [{self.ndi_position_bias[0]:.4f}, "
              f"{self.ndi_position_bias[1]:.4f}, {self.ndi_position_bias[2]:.4f}] {ROBOT_UNIT}")
        print(f"NDI axis scale: [{self.ndi_axis_scale[0]:.6f}, "
              f"{self.ndi_axis_scale[1]:.6f}, {self.ndi_axis_scale[2]:.6f}]")

        t_ee = np.linalg.norm(self.T_ee_marker[:3, 3])
        t_ndi = np.linalg.norm(self.T_ndi_base[:3, 3])
        print(f"\n  [Sanity] ||t_ee_marker||={t_ee:.2f}{ROBOT_UNIT}, "
              f"||t_ndi_base||={t_ndi:.2f}{ROBOT_UNIT}")
        print(f"  [Sanity] det(R_ndi)={np.linalg.det(self.T_ndi_base[:3,:3]):.6f}, "
              f"det(R_ee)={np.linalg.det(self.T_ee_marker[:3,:3]):.6f}\n")

        self.save_calibration_result()

    def calibrate(self):
        print("=" * 60)
        print("2) Eye-to-Hand Calibration")
        print("=" * 60)
        data = self.all_data
        print(f"Using {len(data)} poses\n")

        print("-" * 60)
        print("Fast path: PARK -> PointReg -> NDI axis refinement")
        print("-" * 60)

        T_em = self.solve_handeye_for_T_ee_marker(data)
        T_nb = self.compute_T_ndi_base_from_T_ee_marker(data, T_em)
        pe, re = self.evaluate_absolute_position(data, T_nb, T_em)
        print(f"  PARK: mean={np.mean(pe):.4f}, max={np.max(pe):.4f}{ROBOT_UNIT}")

        T_nb_pr, res_pr = self.solve_point_registration(data, T_ee_marker=T_em)
        pe_pr, re_pr = self.evaluate_absolute_position(data, T_nb_pr, T_em)
        print(f"  PARK+PointReg: mean={np.mean(pe_pr):.4f}, "
              f"max={np.max(pe_pr):.4f}{ROBOT_UNIT}, "
              f"residual_mean={np.mean(res_pr):.4f}{ROBOT_UNIT}")

        T_nb_ax, T_em_ax, ndi_bias_ax, ndi_scale_ax, _ = self.refine_nonlinear_with_ndi_axis_scale(
            data, T_nb_pr, T_em,
            ndi_position_bias_init=np.zeros(3),
            ndi_axis_scale_init=np.ones(3),
            pos_weight=1.0, rot_weight=20.0,
            bias_reg_weight=0.02, scale_reg_weight=10.0)
        pe_ax, re_ax = self.evaluate_absolute_position(
            data, T_nb_ax, T_em_ax,
            ndi_position_bias=ndi_bias_ax,
            ndi_axis_scale=ndi_scale_ax)
        print(f"  PARK+PointReg+NDI-AXIS: mean={np.mean(pe_ax):.4f}, "
              f"max={np.max(pe_ax):.4f}{ROBOT_UNIT}, "
              f"bias=[{ndi_bias_ax[0]:.4f}, {ndi_bias_ax[1]:.4f}, {ndi_bias_ax[2]:.4f}] {ROBOT_UNIT}, "
              f"scale=[{ndi_scale_ax[0]:.6f}, {ndi_scale_ax[1]:.6f}, {ndi_scale_ax[2]:.6f}]")
        print()

        self._set_final_result(
            'PARK+PointReg+NDI-AXIS',
            T_nb_ax, T_em_ax, pe_ax, re_ax,
            ndi_position_bias=ndi_bias_ax,
            ndi_axis_scale=ndi_scale_ax)

    # 9. Validation and diagnostics
    def validate_transform_chain(self, data, T_ndi_base, T_ee_marker,
                                 ndi_position_bias=None, ndi_axis_scale=None):
        """Compare the forward chain with an intentionally inverted wrong chain.

        This is a sanity check to confirm the frame convention and transform order.
        """
        fw_pos, fw_rot = [], []
        inv_pos, inv_rot = [], []

        T_base_ndi = np.linalg.inv(T_ndi_base)
        T_marker_ee = np.linalg.inv(T_ee_marker)

        for _, row in data.iterrows():
            T_ndi_true = self.get_ndi_transform_with_bias(row, ndi_position_bias, ndi_axis_scale)
            T_base_ee = self.get_robot_transform(row)

            T_ndi_pred_fw = T_ndi_base @ T_base_ee @ T_ee_marker
            T_ndi_pred_inv = T_base_ndi @ T_base_ee @ T_marker_ee

            fw_pos.append(np.linalg.norm(T_ndi_pred_fw[:3, 3] - T_ndi_true[:3, 3]))
            inv_pos.append(np.linalg.norm(T_ndi_pred_inv[:3, 3] - T_ndi_true[:3, 3]))

            R_fw = T_ndi_pred_fw[:3, :3].T @ T_ndi_true[:3, :3]
            R_inv = T_ndi_pred_inv[:3, :3].T @ T_ndi_true[:3, :3]

            fw_rot.append(np.degrees(np.arccos(np.clip((np.trace(R_fw) - 1) / 2, -1, 1))))
            inv_rot.append(np.degrees(np.arccos(np.clip((np.trace(R_inv) - 1) / 2, -1, 1))))

        return {
            'forward_pos_mean': float(np.mean(fw_pos)),
            'forward_rot_mean': float(np.mean(fw_rot)),
            'inverse_pos_mean': float(np.mean(inv_pos)),
            'inverse_rot_mean': float(np.mean(inv_rot)),
        }

    def diagnose_root_causes(self, data, T_ndi_base, T_ee_marker,
                             ndi_position_bias=None, ndi_axis_scale=None):
        """Generate simple heuristics for likely calibration error patterns.

        Returns:
            dict with correlation/anisotropy indicators.
        """
        T_base_ndi = np.linalg.inv(T_ndi_base)
        T_marker_ee = np.linalg.inv(T_ee_marker)

        pos_vecs = []
        pos_norms = []
        rot_norms = []
        ndi_dists = []
        robot_dists = []
        idxs = []

        for idx, (_, row) in enumerate(data.iterrows()):
            T_ndi = self.get_ndi_transform_with_bias(row, ndi_position_bias, ndi_axis_scale)
            T_robot_true = self.get_robot_transform(row)
            T_robot_pred = T_base_ndi @ T_ndi @ T_marker_ee

            e_pos = T_robot_pred[:3, 3] - T_robot_true[:3, 3]
            pos_vecs.append(e_pos)
            pos_norms.append(np.linalg.norm(e_pos))

            R_err = T_robot_pred[:3, :3] @ T_robot_true[:3, :3].T
            rv_err = R.from_matrix(R_err).as_rotvec()
            rot_norms.append(np.linalg.norm(rv_err))

            ndi_dists.append(np.linalg.norm(T_ndi[:3, 3]))
            robot_dists.append(np.linalg.norm(T_robot_true[:3, 3]))
            idxs.append(idx)

        pos_vecs = np.array(pos_vecs)
        pos_norms = np.array(pos_norms)
        rot_norms = np.array(rot_norms)
        ndi_dists = np.array(ndi_dists)
        robot_dists = np.array(robot_dists)
        idxs = np.array(idxs, dtype=float)

        def _corr(a, b):
            if len(a) < 2:
                return 0.0
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        axis_std = np.std(pos_vecs, axis=0)
        axis_mean_abs = np.mean(np.abs(pos_vecs), axis=0)
        dominant_axis = int(np.argmax(axis_mean_abs))
        anisotropy = float((np.max(axis_std) + 1e-12) / (np.min(axis_std) + 1e-12))

        # Correlations help detect time drift and geometry-dependent error growth.
        corr_idx = _corr(idxs, pos_norms)
        corr_ndi_dist = _corr(ndi_dists, pos_norms)
        corr_robot_dist = _corr(robot_dists, pos_norms)
        corr_rot_pos = _corr(rot_norms, pos_norms)

        suggestions = []
        if abs(corr_idx) > 0.35:
            suggestions.append('Time-order drift is visible. Check sensor warm-up, mounting repeatability, and fixture stability.')
        if abs(corr_ndi_dist) > 0.35:
            suggestions.append('Error grows with NDI distance. Recheck working distance, field of view, and optical line-of-sight.')
        if anisotropy > 1.8:
            suggestions.append('Error is axis-anisotropic. Recheck TCP direction, camera axis alignment, and robot structural stiffness.')
        if abs(corr_rot_pos) > 0.35:
            suggestions.append('Rotation and position errors are coupled. Recheck tool-tip offset, EE-marker lever arm, and pivot calibration.')
        if not suggestions:
            suggestions.append('No dominant pattern was detected. More diverse poses and distances may improve robustness.')

        return {
            'corr_pose_index_vs_pos': corr_idx,
            'corr_ndi_distance_vs_pos': corr_ndi_dist,
            'corr_robot_distance_vs_pos': corr_robot_dist,
            'corr_rot_vs_pos': corr_rot_pos,
            'axis_std_xyz': axis_std.tolist(),
            'axis_mean_abs_xyz': axis_mean_abs.tolist(),
            'anisotropy_ratio': anisotropy,
            'dominant_axis': ['x', 'y', 'z'][dominant_axis],
            'suggestions': suggestions,
        }

    def validate_all_data(self):
        print("=" * 60)
        print(f"3) Full-Dataset Validation (absolute position error, unit: {ROBOT_UNIT})")
        print("=" * 60)

        pe, re = self.evaluate_absolute_position(
            self.all_data, self.T_ndi_base, self.T_ee_marker,
            ndi_position_bias=self.ndi_position_bias,
            ndi_axis_scale=self.ndi_axis_scale)

        print(f"\n{'Pose':<8} {'Pos Err(mm)':<15} {'Rot Err(deg)':<15}")
        print("-" * 38)
        for i, (p, r_) in enumerate(zip(pe, re)):
            flag = " ***" if p > 1.0 else ""
            print(f"{i:<8} {p:<15.4f} {r_:<15.4f}{flag}")

        print(f"\nPosition: mean={np.mean(pe):.4f}, std={np.std(pe):.4f}, "
              f"min={np.min(pe):.4f}, max={np.max(pe):.4f}{ROBOT_UNIT}")
        print(f"Rotation: mean={np.mean(re):.4f} deg, std={np.std(re):.4f} deg")

        chain = self.validate_transform_chain(
            self.all_data, self.T_ndi_base, self.T_ee_marker,
            ndi_position_bias=self.ndi_position_bias,
            ndi_axis_scale=self.ndi_axis_scale)
        print("\n[Transform Chain Check]")
        print(f"  Forward model (T_ndi_base @ T_base_ee @ T_ee_marker)"
              f": pos={chain['forward_pos_mean']:.4f}{ROBOT_UNIT}, "
              f"rot={chain['forward_rot_mean']:.4f} deg")
        print(f"  Inverse model (inv(T_ndi_base) @ T_base_ee @ inv(T_ee_marker))"
              f": pos={chain['inverse_pos_mean']:.4f}{ROBOT_UNIT}, "
              f"rot={chain['inverse_rot_mean']:.4f} deg")

        diag = self.diagnose_root_causes(
            self.all_data, self.T_ndi_base, self.T_ee_marker,
            ndi_position_bias=self.ndi_position_bias,
            ndi_axis_scale=self.ndi_axis_scale)
        print("\n[Root Cause Hints]")
        print(f"  corr(pose_index, pos_err)={diag['corr_pose_index_vs_pos']:.3f}")
        print(f"  corr(||t_ndi||, pos_err)={diag['corr_ndi_distance_vs_pos']:.3f}")
        print(f"  corr(||t_robot||, pos_err)={diag['corr_robot_distance_vs_pos']:.3f}")
        print(f"  corr(rot_err, pos_err)={diag['corr_rot_vs_pos']:.3f}")
        print(f"  axis_std xyz={np.array(diag['axis_std_xyz'])}")
        print(f"  anisotropy={diag['anisotropy_ratio']:.3f}, dominant_axis={diag['dominant_axis']}")
        for s_msg in diag['suggestions']:
            print(f"   - {s_msg}")

        target_met = np.mean(pe) < 1.0
        print(f"\nTarget (mean < 1.0 mm): {'YES' if target_met else 'NO'} "
              f"(current: {np.mean(pe):.4f}{ROBOT_UNIT})\n")
        return pe, re

    # 10. Result export
    def save_calibration_result(self):
        output_dir  = Path('../dataset/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        euler_ndi = R.from_matrix(
            self.T_ndi_base[:3, :3]).as_euler('ZYX', degrees=True)
        euler_ee = R.from_matrix(
            self.T_ee_marker[:3, :3]).as_euler('ZYX', degrees=True)

        result = {
            'T_ndi_base': {
                'matrix': self.T_ndi_base.tolist(),
                'translation_vector': self.T_ndi_base[:3, 3].tolist(),
                'euler_ZYX_deg': {
                    'Z': float(euler_ndi[0]), 'Y': float(euler_ndi[1]),
                    'X': float(euler_ndi[2]),
                },
                'description': 'NDI camera frame -> Robot base frame',
            },
            'T_ee_marker': {
                'matrix': self.T_ee_marker.tolist(),
                'translation_vector': self.T_ee_marker[:3, 3].tolist(),
                'euler_ZYX_deg': {
                    'Z': float(euler_ee[0]), 'Y': float(euler_ee[1]),
                    'X': float(euler_ee[2]),
                },
                'description': 'Robot EE frame -> Marker frame',
            },
            'ndi_position_bias': self.ndi_position_bias.tolist(),
            'ndi_axis_scale': self.ndi_axis_scale.tolist(),
            'translation_unit': ROBOT_UNIT,
            'method': self.method_name,
            'num_poses': len(self.all_data),
            'navigation_formula':
                'T_base_ee_target = inv(T_ndi_base) @ T_ndi_marker_new @ inv(T_ee_marker)',
        }
        output_path = output_dir / 'calibration_result.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {output_path}\n")

    # 11. Runtime helper
    def predict_ee_from_ndi(self, T_ndi_marker_new):
        """Convert a new NDI marker pose into a target robot EE pose.

        Returns: (T_base_ee_target, p_target [x,y,z], euler_target [u,v,w])
        """
        T_base_ndi = np.linalg.inv(self.T_ndi_base)
        T_marker_ee = np.linalg.inv(self.T_ee_marker)
        T_ndi_marker_corr = T_ndi_marker_new.copy()
        T_ndi_marker_corr[:3, 3] = self.ndi_axis_scale * T_ndi_marker_corr[:3, 3] + self.ndi_position_bias
        T_target = T_base_ndi @ T_ndi_marker_corr @ T_marker_ee

        p = T_target[:3, 3]
        zyx = R.from_matrix(T_target[:3, :3]).as_euler('ZYX', degrees=True)
        euler = np.array([zyx[2], zyx[1], zyx[0]])  # [u, v, w]
        return T_target, p, euler

    # 12. Visualization
    def visualize_results(self, pos_errors, rot_errors):
        print("=" * 60)
        print("4) Visualization")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.bar(range(len(pos_errors)), pos_errors, alpha=0.7, color='steelblue')
        ax.axhline(np.mean(pos_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(pos_errors):.4f} {ROBOT_UNIT}')
        ax.axhline(1.0, color='orange', linestyle=':', label='Target: 1.0 mm')
        ax.set_xlabel('Pose Index')
        ax.set_ylabel(f'Position Error ({ROBOT_UNIT})')
        ax.set_title('Absolute Position Error per Pose')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.bar(range(len(rot_errors)), rot_errors, alpha=0.7, color='seagreen')
        ax.axhline(np.mean(rot_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rot_errors):.4f} deg')
        ax.set_xlabel('Pose Index')
        ax.set_ylabel('Rotation Error (deg)')
        ax.set_title('Absolute Rotation Error per Pose')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.hist(pos_errors, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(1.0, color='orange', linestyle=':', label='Target: 1.0 mm')
        ax.set_xlabel(f'Position Error ({ROBOT_UNIT})')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Error Distribution')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')
        summary_lines = [
            f"Method: {self.method_name}",
            f"Poses: {len(self.all_data)}",
            f"Pos mean: {np.mean(pos_errors):.4f}{ROBOT_UNIT}",
            f"Pos max: {np.max(pos_errors):.4f}{ROBOT_UNIT}",
            f"Rot mean: {np.mean(rot_errors):.4f} deg",
            f"NDI bias: [{self.ndi_position_bias[0]:.3f}, {self.ndi_position_bias[1]:.3f}, {self.ndi_position_bias[2]:.3f}]",
            f"NDI scale: [{self.ndi_axis_scale[0]:.5f}, {self.ndi_axis_scale[1]:.5f}, {self.ndi_axis_scale[2]:.5f}]",
        ]
        ax.set_title('Calibration Summary')
        ax.text(0.02, 0.98, "\n".join(summary_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

        config = (f"Method: {self.method_name} | Poses: {len(self.all_data)} | "
                  f"Pos mean: {np.mean(pos_errors):.4f}{ROBOT_UNIT} | "
                  f"Rot mean: {np.mean(rot_errors):.4f} deg")
        fig.text(0.5, 0.01, config, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out = Path('./dataset/results/calibration_visualization.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {out}")
        plt.close()
        print()

    # 13. End-to-end runner
    def run(self):
        self.load_and_preprocess_data()
        self.calibrate()
        pe, re = self.validate_all_data()
        self.visualize_results(pe, re)
        print("=" * 60)
        print("Calibration complete.")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '../dataset/calibration/calibration_data.csv'
    print(f"Input file: {csv_path}\n")
    calibration = HandEyeCalibration(csv_path=csv_path)
    calibration.run()



