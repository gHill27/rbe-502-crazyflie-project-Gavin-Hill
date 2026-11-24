"""Geometric tracking controller for the Crazyflie 2.X.

RBE 502 Quadrotor Control Project.

Complete the three blocks marked "YOUR CODE STARTS HERE":
    1. the gain matrices in `__init__`,
    2. `_positionControl` (outer loop),
    3. `_attitudeControl` (inner loop).
Do not change the function signatures or the rest of this file. We grade
your controller by running this file with the provided code.

Test `_attitudeControl` on its own with `examples/hover_test.py` before you
track the trajectories.

Reference: T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control
of a quadrotor UAV on SE(3)," IEEE CDC, 2010. The paper points e3 downward,
while this project and the simulator point it upward.
"""
import numpy as np #type: ignore
import pybullet as p #type: ignore

from gym_pybullet_drones.control.BaseControl import BaseControl #type: ignore
from gym_pybullet_drones.utils.enums import DroneModel #type: ignore


def hat(a):
    """Hat map: returns the skew-symmetric matrix of `a`, so that hat(a) @ b = a x b."""
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def vee(S):
    """Vee map: the inverse of the hat map, returns the vector of a skew-symmetric matrix `S`."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class GeometricControl(BaseControl):
    """Geometric tracking controller on SE(3) for the Crazyflie 2.X (X configuration)."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 g: float = 9.8
                 ):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X:
            raise ValueError("GeometricControl only supports DroneModel.CF2X.")

        #### Quadrotor parameters, read from cf2x.urdf (do not change) ###########
        self.g = g                                          # gravitational acceleration [m/s^2]
        self.m = self._getURDFParameter('m')                # mass [kg]
        self.J = np.diag([self._getURDFParameter('ixx'),
                          self._getURDFParameter('iyy'),
                          self._getURDFParameter('izz')])   # inertia matrix about the body axes [kg m^2]
        self.e3 = np.array([0.0, 0.0, 1.0])

        ##########################################################################
        #### YOUR CODE STARTS HERE: set the controller gains #####################
        # Use positive-definite diagonal matrices, for example np.diag([1.0, 1.0, 1.0]).
        # self.K_x = np.diag((0, 0, 0))         # position gain
        # self.K_v = np.diag((0, 0, 0))         # velocity gain
        self.K_x = np.diag((0.12, 0.12, 0.12))         # position gain
        self.K_v = np.diag((0.24, 0.24, 0.24))         # velocity gain
        self.K_R = np.diag([0.005, 0.005, 0.001])         # attitude gain
        self.K_Omega = np.diag([0.001, 0.001, 0.0002])     # angular velocity gain
        #### YOUR CODE ENDS HERE #################################################
        ##########################################################################

        #### Control allocation (do not change) ##################################
        d = self._getURDFParameter('arm') / np.sqrt(2)      # rotor offset along b1 and b2 [m]
        gamma = self.KM / self.KF                           # reaction moment to thrust ratio [m]
        # [f, M1, M2, M3] = ALLOCATION @ [F1, F2, F3, F4], see the project statement.
        # Rotor i (1 to 4) in the statement is entry i-1 of the rotor arrays in the code.
        self.ALLOCATION = np.array([[1.0, 1.0, 1.0, 1.0],
                                    [-d, -d, d, d],
                                    [-d, d, d, -d],
                                    [-gamma, gamma, -gamma, gamma]])
        self.ALLOCATION_INV = np.linalg.inv(self.ALLOCATION)
        self.MAX_ROTOR_THRUST = self._getURDFParameter('thrust2weight') * self.m * self.g / 4  # [N]

    ################################################################################

    def computeControlFromState(self,
                                state,
                                target_pos,
                                target_vel=np.zeros(3),
                                target_acc=np.zeros(3),
                                target_yaw=0.0
                                ):
        """Computes the rotor speeds from the state vector returned by the environment.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped state vector from `CtrlAviary`: position (0:3), quaternion
            [qx, qy, qz, qw] (3:7), roll-pitch-yaw (7:10), velocity (10:13), angular
            velocity in the inertial frame (13:16), and last action (16:20).
        target_pos, target_vel, target_acc : ndarray
            (3,)-shaped desired position, velocity, and acceleration.
        target_yaw : float
            Desired yaw angle [rad].

        Returns
        -------
        ndarray
            (4,)-shaped rotor speeds [rpm].
        dict
            The total thrust, moment, rotation matrices, and body angular velocity,
            used for logging.

        """
        return self.computeControl(cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel_world=state[13:16],
                                   target_pos=target_pos,
                                   target_vel=target_vel,
                                   target_acc=target_acc,
                                   target_yaw=target_yaw
                                   )

    ################################################################################

    def computeAttitudeControlFromState(self,
                                        state,
                                        thrust,
                                        target_rot=np.eye(3)
                                        ):
        """Runs only the attitude controller and the control allocation (hover test).

        The position controller is skipped: the total thrust and the desired attitude
        are given, and the desired angular velocity is zero. `hover_test.py` uses this
        function to test `_attitudeControl` on its own.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped state vector from `CtrlAviary`, as in `computeControlFromState`.
        thrust : float
            Total thrust f [N].
        target_rot : ndarray
            (3, 3)-shaped desired rotation matrix R_d.

        Returns
        -------
        ndarray
            (4,)-shaped rotor speeds [rpm].
        dict
            The total thrust, moment, rotation matrices, and body angular velocity,
            used for logging.

        """
        self.control_counter += 1
        _, _, R, Omega = self._convertState(state[0:3], state[3:7], state[10:13], state[13:16])
        return self._innerLoopAndAllocation(R, Omega, float(thrust), np.asarray(target_rot, dtype=float))

    ################################################################################

    def computeControl(self,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel_world,
                       target_pos,
                       target_vel,
                       target_acc,
                       target_yaw
                       ):
        """Runs the position and attitude controllers and the control allocation."""
        self.control_counter += 1
        x, v, R, Omega = self._convertState(cur_pos, cur_quat, cur_vel, cur_ang_vel_world)

        #### Outer loop: total thrust and desired attitude #######################
        thrust, target_rot = self._positionControl(x, v, R,
                                                   np.asarray(target_pos, dtype=float),
                                                   np.asarray(target_vel, dtype=float),
                                                   np.asarray(target_acc, dtype=float),
                                                   float(target_yaw))
        if np.size(thrust) != 1:
            raise ValueError("_positionControl must return the total thrust as a scalar.")
        if np.shape(target_rot) != (3, 3):
            raise ValueError("_positionControl must return the desired rotation matrix as a (3, 3) array.")
        thrust = float(np.squeeze(thrust))
        target_rot = np.asarray(target_rot, dtype=float)

        #### Inner loop and control allocation ###################################
        return self._innerLoopAndAllocation(R, Omega, thrust, target_rot)

    ################################################################################

    def _convertState(self, cur_pos, cur_quat, cur_vel, cur_ang_vel_world):
        """Converts the simulator state into x, v, R, and Omega."""
        x = np.asarray(cur_pos, dtype=float)
        v = np.asarray(cur_vel, dtype=float)
        # PyBullet quaternions are [qx, qy, qz, qw]. The columns of R are b1, b2, b3.
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # PyBullet reports the angular velocity in the inertial frame; convert it to the body frame.
        Omega = R.T @ np.asarray(cur_ang_vel_world, dtype=float)
        return x, v, R, Omega

    ################################################################################

    def _innerLoopAndAllocation(self, R, Omega, thrust, target_rot):
        """Runs the attitude controller and converts the thrust and moment into rotor speeds."""
        #### Inner loop: moment ##################################################
        moment = self._attitudeControl(R, Omega, target_rot)
        if np.size(moment) != 3:
            raise ValueError("_attitudeControl must return the moment as a (3,) array.")
        moment = np.asarray(moment, dtype=float).reshape(3)

        #### Control allocation ##################################################
        rotor_thrusts = np.clip(self.ALLOCATION_INV @ np.hstack([thrust, moment]), 0.0, self.MAX_ROTOR_THRUST)
        rpm = np.sqrt(rotor_thrusts / self.KF)

        info = {'thrust': thrust,
                'moment': moment,
                'rot': R,
                'target_rot': target_rot,
                'ang_vel': Omega,
                'rotor_thrusts': rotor_thrusts}
        return rpm, info

    ################################################################################

    def _positionControl(self,
                         cur_pos,
                         cur_vel,
                         cur_rot,
                         target_pos,
                         target_vel,
                         target_acc,
                         target_yaw
                         ):
        """Position controller (outer loop).

        Parameters
        ----------
        cur_pos : ndarray
            (3,)-shaped position x of the center of mass in the inertial frame [m].
        cur_vel : ndarray
            (3,)-shaped velocity v in the inertial frame [m/s].
        cur_rot : ndarray
            (3, 3)-shaped rotation matrix R, whose columns are b1, b2, b3 in the inertial frame.
        target_pos, target_vel, target_acc : ndarray
            (3,)-shaped desired position x_d [m], velocity v_d [m/s], and acceleration a_d [m/s^2].
        target_yaw : float
            Desired yaw angle psi_d [rad].

        Returns
        -------
        float
            Total thrust f [N].
        ndarray
            (3, 3)-shaped desired rotation matrix R_d.

        """
        ##########################################################################
        #### YOUR CODE STARTS HERE ###############################################
        # thrust = 0.0
        e_x = cur_pos - target_pos 
        e_v = cur_vel - target_vel
        Fd = -self.K_x@e_x - self.K_v@e_v + self.m*self.g*self.e3 + self.m*target_acc
        thrust = Fd@cur_rot@self.e3
        b3 = Fd/np.linalg.norm(Fd)
        b1c = np.array([np.cos(target_yaw), np.sin(target_yaw),0])
        term =  np.cross(b3,b1c)
        b2 = term/np.linalg.norm(term)
        b1 = np.cross(b2,b3)
        target_rot = np.column_stack([b1,b2,b3])
        # target_rot = np.eye(3)
        #### YOUR CODE ENDS HERE #################################################
        ##########################################################################
        return thrust, target_rot, Fd

    ################################################################################

    def _attitudeControl(self,
                         cur_rot,
                         cur_ang_vel,
                         target_rot,
                         target_omega
                         ):
        """Attitude controller (inner loop), with Omega_d = 0.

        Parameters
        ----------
        cur_rot : ndarray
            (3, 3)-shaped rotation matrix R.
        cur_ang_vel : ndarray
            (3,)-shaped angular velocity Omega in the body frame [rad/s].
        target_rot : ndarray
            (3, 3)-shaped desired rotation matrix R_d from `_positionControl`.

        Returns
        -------
        ndarray
            (3,)-shaped moment M about the body axes [N m].

        """
        ##########################################################################
        #### YOUR CODE STARTS HERE ###############################################
        e_r = 0.5 * vee(target_rot.T @ cur_rot - cur_rot.T @ target_rot)
        w = cur_rot.T @ target_rot @ target_omega            # R^T R_d Omega_d
        e_omega = cur_ang_vel - w
        moment = (-self.K_R @ e_r - self.K_Omega @ e_omega
                  + np.cross(cur_ang_vel, self.J @ cur_ang_vel)
                  - self.J @ np.cross(cur_ang_vel, w))       # Omega_d_dot = 0
        return moment
        #### YOUR CODE ENDS HERE #################################################
        ##########################################################################
        return moment

    def _desiredOmega(self, R_d, F_d, jerk, yaw, yawdot):
        """Omega_d from differential flatness, expressed in the desired body frame."""
        b1, b2, b3 = R_d[:, 0], R_d[:, 1], R_d[:, 2]
        f = np.linalg.norm(F_d)
        W1 = -self.m * (jerk @ b2) / f
        W2 =  self.m * (jerk @ b1) / f
        b1c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        e2c = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        W3 = (W1 * (b3 @ b1c) + yawdot * (b2 @ e2c)) / (b1 @ b1c)
        return np.array([W1, W2, W3])