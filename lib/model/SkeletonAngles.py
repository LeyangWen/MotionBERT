# a class for vicon skeleton
from unicodedata import category

import numpy as np
from ergo3d.geometryPytorch import Point, Plane, CoordinateSystem3D, JointAngles
import torch

# todo: self.frame_no vs self.frame_number be consistent

class SkeletonAngles:
    def __init__(self):
        pass

    def load_name_list(self, name_list):
        self.point_labels = name_list
        self.point_number = len(name_list)

    def load_points(self, pt):
        """
        pt in tensor        # torch.Size([1, 16384, 22, 3])
        """
        try:
            self.point_labels
        except(AttributeError):
            print('point_labels is empty, need to load point names first')
            raise AttributeError
        if len(pt.shape) == 4:
            pt = pt.squeeze(0)
        self.frame_number, _, self.points_dimension = pt.shape
        self.point_poses = {}
        for i in range(self.point_number):
            if self.points_dimension == 3:
                xyz = pt[:, i, :]
                self.point_poses[self.point_labels[i]] = Point(xyz, name=self.point_labels[i])

class VEHS37SkeletonAngles(SkeletonAngles):
    def __init__(self, kpt_xyz):
        """
        kpt_xyz: torch.Size([N, T, 37, 3])

        """
        super().__init__()
        rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER', 'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5', 'LMCP2', 'LMCP5']
        if len(kpt_xyz.shape) != 4:
            raise ValueError(f'kpt_xyz should be in shape of (N, T, 37, 3), but got {kpt_xyz.shape}')
        self.N = kpt_xyz.shape[0]
        self.T = kpt_xyz.shape[1]
        self.kpt = kpt_xyz.shape[2]
        kpt_xyz = kpt_xyz.reshape(self.N*self.T, 37, 3)
        self.load_name_list(rtm_pose_37_keypoints_vicon_dataset_v1)
        self.load_points(kpt_xyz)
        self.angle_names = ['neck', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist', 'back', 'right_knee', 'left_knee']
        self.sub_angle_names = ['flexion', 'abduction', 'rotation']
    
    def get_angles(self):
        """
        Output: (N, T, angle_no) in pytorch tensor
        """
        all_angles = None
        for angle_index, this_angle_name in enumerate(self.angle_names):
            joint = getattr(self, this_angle_name + '_angles')()
            for this_ergo_angle in self.sub_angle_names:
                angle = getattr(joint, this_ergo_angle)
                if angle is not None:
                    angle = angle.unsqueeze(1)
                    if all_angles is None:
                        all_angles = angle
                    else:
                        all_angles = torch.cat((all_angles, angle), dim=1)
        return all_angles.reshape(self.N, self.T, -1)

    def neck_angles(self):
        zero_frame = [-90, -180, -180]
        REAR = self.point_poses['REAR']
        LEAR = self.point_poses['LEAR']
        HDTP = self.point_poses['HDTP']
        EAR = Point.mid_point(REAR, LEAR)
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        C7 = self.point_poses['C7']
        # RPSIS = self.point_poses['RPSIS']
        # LPSIS = self.point_poses['LPSIS']
        PELVIS = self.point_poses['PELVIS']

        HEAD_plane = Plane()
        HEAD_plane.set_by_pts(REAR, LEAR, HDTP)
        HEAD_coord = CoordinateSystem3D()
        HEAD_coord.set_by_plane(HEAD_plane, EAR, HDTP, sequence='yxz', axis_positive=True)
        NECK_angles = JointAngles()
        NECK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-bend', 'rotation': 'rotation'}  # lateral bend
        NECK_angles.set_zero(zero_frame, by_frame=False)
        NECK_angles.get_flex_abd(HEAD_coord, Point.vector(C7, PELVIS), plane_seq=['xy', 'yz'], flip_sign=[1, -1])
        NECK_angles.get_rot(LEAR, REAR, LSHOULDER, RSHOULDER)
        return NECK_angles

    def right_shoulder_angles(self):
        zero_frame = [0, 90, 90]
        RSHOULDER = self.point_poses['RSHOULDER']
        C7 = self.point_poses['C7']
        C7_d = self.point_poses['C7_d']
        # PELVIS_b = Point.mid_point(self.point_poses['RPSIS'], self.point_poses['LPSIS'])
        PELVIS = self.point_poses['PELVIS']
        # C7_m = self.point_poses['C7_m']
        SS = self.point_poses['SS']
        RELBOW = self.point_poses['RELBOW']
        RAP_b = self.point_poses['RAP_b']
        RAP_f = self.point_poses['RAP_f']
        RME = self.point_poses['RME']
        RLE = self.point_poses['RLE']

        RSHOULDER_plane = Plane()
        RSHOULDER_plane.set_by_vector(RSHOULDER, Point.vector(C7_d, PELVIS), direction=-1)
        # RSHOULDER_C7_m_project = RSHOULDER_plane.project_point(C7_m)
        RSHOULDER_SS_project = RSHOULDER_plane.project_point(SS)
        RSHOULDER_coord = CoordinateSystem3D()
        RSHOULDER_coord.set_by_plane(RSHOULDER_plane, C7_d, RSHOULDER_SS_project, sequence='xyz', axis_positive=True)  # new: use back to chest vector
        # RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, RSHOULDER_C7_m_project, sequence='zyx', axis_positive=False)  # old: use shoulder to chest vector
        RSHOULDER_angles = JointAngles()
        RSHOULDER_angles.set_zero(zero_frame, by_frame=False)
        RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, Point.vector(RSHOULDER, RELBOW), plane_seq=['xy', 'xz'])
        RSHOULDER_angles.get_rot(RAP_b, RAP_f, RME, RLE)

        if False:  # shoulder angles used in paper
            RSHOULDER_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'abduction', 'rotation': 'rotation'}
        else:
            RSHOULDER_angles.ergo_name = {'flexion': 'elevation', 'abduction': 'H-abduction', 'rotation': 'rotation'}
            RSHOULDER_angles.flexion = Point.angle(Point.vector(RSHOULDER, RELBOW).xyz, Point.vector(C7, PELVIS).xyz)
            RSHOULDER_angles.flexion = RSHOULDER_angles.zero_by_idx(0)  # zero by zero frame after setting flexion without function

            shoulder_threshold = 10/180*np.pi  # the H-abduction is not well defined when the flexion is small or near 180 degrees
            shoulder_filter = torch.logical_and(torch.abs(RSHOULDER_angles.flexion) > shoulder_threshold,
                                torch.abs(RSHOULDER_angles.flexion) < (np.pi - shoulder_threshold))
            RSHOULDER_angles.abduction = torch.where(
                                        shoulder_filter, RSHOULDER_angles.abduction, torch.zeros_like(RSHOULDER_angles.abduction)
)
        return RSHOULDER_angles

    def left_shoulder_angles(self):     # not checked
        zero_frame = [0, 0, -90]
        LSHOULDER = self.point_poses['LSHOULDER']
        C7 = self.point_poses['C7']
        C7_d = self.point_poses['C7_d']
        # PELVIS_b = Point.mid_point(self.point_poses['RPSIS'], self.point_poses['LPSIS'])
        PELVIS = self.point_poses['PELVIS']
        # C7_m = self.point_poses['C7_m']
        SS = self.point_poses['SS']
        LELBOW = self.point_poses['LELBOW']
        LAP_b = self.point_poses['LAP_b']
        LAP_f = self.point_poses['LAP_f']
        LME = self.point_poses['LME']
        LLE = self.point_poses['LLE']

        LSHOULDER_plane = Plane()
        LSHOULDER_plane.set_by_vector(LSHOULDER, Point.vector(C7_d, PELVIS), direction=-1)
        # LSHOULDER_C7_m_project = LSHOULDER_plane.project_point(C7_m)
        LSHOULDER_SS_project = LSHOULDER_plane.project_point(SS)
        LSHOULDER_coord = CoordinateSystem3D()
        LSHOULDER_coord.set_by_plane(LSHOULDER_plane, C7_d, LSHOULDER_SS_project, sequence='zyx', axis_positive=True)
        LSHOULDER_angles = JointAngles()
        LSHOULDER_angles.get_flex_abd(LSHOULDER_coord, Point.vector(LSHOULDER, LELBOW), plane_seq=['xy', 'xz'], flip_sign=[1, -1])
        LSHOULDER_angles.set_zero(zero_frame, by_frame=False)
        LSHOULDER_angles.get_rot(LAP_b, LAP_f, LME, LLE)
        if False:  # shoulder angles used in paper
            LSHOULDER_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'abduction', 'rotation': 'rotation'}  # horizontal abduction
        else:  # shoulder angles used in VEHS application
            LSHOULDER_angles.ergo_name = {'flexion': 'elevation', 'abduction': 'H-abduction', 'rotation': 'rotation'}  # horizontal abduction
            LSHOULDER_angles.flexion = Point.angle(Point.vector(LSHOULDER, LELBOW).xyz, Point.vector(C7, PELVIS).xyz)
            LSHOULDER_angles.flexion = LSHOULDER_angles.zero_by_idx(0)
            shoulder_threshold = 10/180*np.pi
            shoulder_filter = torch.logical_and(
                torch.abs(LSHOULDER_angles.flexion) > shoulder_threshold,
                torch.abs(LSHOULDER_angles.flexion) < (np.pi - shoulder_threshold)
            )

            LSHOULDER_angles.abduction = torch.where(
                shoulder_filter, 
                LSHOULDER_angles.abduction, 
                torch.zeros_like(LSHOULDER_angles.abduction)
            )
        return LSHOULDER_angles

    def right_elbow_angles(self):
        zero_frame = [-180, 0, 0]
        RELBOW = self.point_poses['RELBOW']
        RSHOULDER = self.point_poses['RSHOULDER']
        RWRIST = self.point_poses['RWRIST']

        RELBOW_angles = JointAngles()
        RELBOW_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        RELBOW_angles.set_zero(zero_frame, by_frame=False)
        RELBOW_angles.flexion = -Point.angle(Point.vector(RELBOW, RSHOULDER).xyz, Point.vector(RELBOW, RWRIST).xyz)
        RELBOW_angles.flexion = RELBOW_angles.zero_by_idx(0)  # zero by zero frame
        RELBOW_angles.is_empty = False
        RELBOW_angles.abduction = None
        RELBOW_angles.rotation = None
        return RELBOW_angles

    def left_elbow_angles(self):  # not checked
        zero_frame = [-180, 0, 0]
        LELBOW = self.point_poses['LELBOW']
        LSHOULDER = self.point_poses['LSHOULDER']
        LWRIST = self.point_poses['LWRIST']

        LELBOW_angles = JointAngles()
        LELBOW_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        LELBOW_angles.set_zero(zero_frame, by_frame=False)
        LELBOW_angles.flexion = -Point.angle(Point.vector(LELBOW, LSHOULDER).xyz, Point.vector(LELBOW, LWRIST).xyz)
        LELBOW_angles.flexion = LELBOW_angles.zero_by_idx(0)
        LELBOW_angles.is_empty = False
        LELBOW_angles.abduction = None
        LELBOW_angles.rotation = None
        return LELBOW_angles

    def right_wrist_angles(self):
        zero_frame = [-90, -180, -90]
        RWRIST = self.point_poses['RWRIST']
        RMCP2 = self.point_poses['RMCP2']
        RMCP5 = self.point_poses['RMCP5']
        RHAND = self.point_poses['RHAND']
        try:
            RRS = self.point_poses['RRS']
            RUS = self.point_poses['RUS']
        except:
            pass
        RLE = self.point_poses['RLE']
        RME = self.point_poses['RME']
        RELBOW = self.point_poses['RELBOW']

        RWRIST_plane = Plane()
        RWRIST_plane.set_by_pts(RMCP2, RWRIST, RMCP5)
        RWRIST_coord = CoordinateSystem3D()
        RWRIST_coord.set_by_plane(RWRIST_plane, RWRIST, RHAND, sequence='yxz', axis_positive=True)
        RWRIST_angles = JointAngles()
        RWRIST_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'deviation', 'rotation': 'pronation'}
        RWRIST_angles.set_zero(zero_frame, by_frame=False)
        RWRIST_angles.get_flex_abd(RWRIST_coord, Point.vector(RWRIST, RELBOW), plane_seq=['xy', 'yz'])
        try:
            RWRIST_angles.get_rot(RRS, RUS, RLE, RME)
        except:
            RWRIST_angles.get_rot(RMCP2, RMCP5, RLE, RME)
        return RWRIST_angles

    def left_wrist_angles(self):  # not checked
        zero_frame = [-90, -180, 90]
        LWrist = self.point_poses['LWRIST']
        LMCP2 = self.point_poses['LMCP2']
        LMCP5 = self.point_poses['LMCP5']  # todo: more accurate using LRS
        LHand = self.point_poses['LHAND']
        try:
            LRS = self.point_poses['LRS']
            LUS = self.point_poses['LUS']
        except:
            pass
        LLE = self.point_poses['LLE']
        LME = self.point_poses['LME']
        LELBOW = self.point_poses['LELBOW']

        LWrist_plane = Plane()
        LWrist_plane.set_by_pts(LMCP2, LWrist, LMCP5)
        LWrist_coord = CoordinateSystem3D()
        LWrist_coord.set_by_plane(LWrist_plane, LWrist, LHand, sequence='yxz', axis_positive=True)
        LWrist_angles = JointAngles()
        LWrist_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'deviation ', 'rotation': 'pronation'}
        LWrist_angles.set_zero(zero_frame, by_frame=False)
        LWrist_angles.get_flex_abd(LWrist_coord, Point.vector(LWrist, LELBOW), plane_seq=['xy', 'yz'])
        try:
            LWrist_angles.get_rot(LRS, LUS, LLE, LME)
        except:
            LWrist_angles.get_rot(LMCP2, LMCP5, LLE, LME)
        return LWrist_angles

    def back_angles(self, up_axis=[0, 1000, 0], zero_frame = [-90, 180, 180]):
        # todo: back correction
        C7 = self.point_poses['C7']
        # RPSIS = self.point_poses['RPSIS']
        # LPSIS = self.point_poses['LPSIS']
        RHIP = self.point_poses['RHIP']
        LHIP = self.point_poses['LHIP']
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        # PELVIS_b = Point.mid_point(RPSIS, LPSIS)
        PELVIS = self.point_poses['PELVIS']

        BACK_plane = Plane()
        BACK_plane.set_by_vector(PELVIS, Point.create_const_vector(*up_axis, examplePt=PELVIS), direction=1)
        BACK_coord = CoordinateSystem3D()
        # BACK_RPSIS_PROJECT = BACK_plane.project_point(RPSIS)
        BACK_RHIP_PROJECT = BACK_plane.project_point(RHIP)
        BACK_coord.set_by_plane(BACK_plane, PELVIS, BACK_RHIP_PROJECT, sequence='zyx', axis_positive=True)
        BACK_angles = JointAngles()
        BACK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-flexion', 'rotation': 'rotation'}  #lateral flexion
        BACK_angles.set_zero(zero_frame)
        BACK_angles.get_flex_abd(BACK_coord, Point.vector(PELVIS, C7), plane_seq=['xy', 'yz'])
        # BACK_angles.get_rot(RSHOULDER, LSHOULDER, RPSIS, LPSIS, flip_sign=1)
        BACK_angles.get_rot(RSHOULDER, LSHOULDER, RHIP, LHIP, flip_sign=1)

        return BACK_angles

    def right_knee_angles(self):
        zero_frame = -180
        RKNEE = self.point_poses['RKNEE']
        RHIP = self.point_poses['RHIP']
        RANKLE = self.point_poses['RANKLE']

        RKNEE_angles = JointAngles()
        RKNEE_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        RKNEE_angles.set_zero(zero_frame, by_frame=False)
        RKNEE_angles.flexion = -Point.angle(Point.vector(RKNEE, RHIP).xyz, Point.vector(RKNEE, RANKLE).xyz)
        RKNEE_angles.flexion = RKNEE_angles.zero_by_idx(0)  # zero by zero frame
        RKNEE_angles.is_empty = False
        RKNEE_angles.abduction = None
        RKNEE_angles.rotation = None
        return RKNEE_angles

    def left_knee_angles(self):  # not checked
        zero_frame = -180
        LKNEE = self.point_poses['LKNEE']
        LHIP = self.point_poses['LHIP']
        LANKLE = self.point_poses['LANKLE']

        LKNEE_angles = JointAngles()
        LKNEE_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        LKNEE_angles.set_zero(zero_frame, by_frame=False)
        LKNEE_angles.flexion = -Point.angle(Point.vector(LKNEE, LHIP).xyz, Point.vector(LKNEE, LANKLE).xyz)
        LKNEE_angles.flexion = LKNEE_angles.zero_by_idx(0)
        LKNEE_angles.is_empty = False
        LKNEE_angles.abduction = None
        LKNEE_angles.rotation = None
        return LKNEE_angles