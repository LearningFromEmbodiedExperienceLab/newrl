"""Name/regex selection, ordering, and bound observation/reward subsets."""

from types import SimpleNamespace
import unittest

import torch
import warp as wp
from tensordict import TensorDict

from newrl.scene import RobotEntity
from newrl.mdp.observations import BodyHeight, BodyLinearVelocity, EntityField
from newrl.mdp.rewards import JointLimit, JointVel2, TrackBodyMotion
from newrl.utils.string import resolve_matching_names, resolve_matching_names_values, resolve_scene_names


class NameMatchingTests(unittest.TestCase):
    def test_g1_config_patterns_preserve_the_original_tracking_bodies(self):
        from newrl.algos.supermimic import SuperMimicConfig
        from newrl.configs import G1MotionTracking
        from newrl.assets.robots import assetx_g1

        names = assetx_g1().build().body_label
        original = ('pelvis','left_hip_roll_link','left_knee_link','left_ankle_roll_link',
                    'right_hip_roll_link','right_knee_link','right_ankle_roll_link','torso_link',
                    'left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link',
                    'right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link')
        expected = resolve_scene_names(original,names)
        self.assertEqual(len(expected),14)
        task = G1MotionTracking()
        for name in ('body_pos','body_ori','body_lin_vel','body_ang_vel'):
            self.assertEqual(resolve_scene_names(dict(task.mdp.rewards['tracking'].terms)[name].body_names,names),expected)
        self.assertEqual(resolve_scene_names(SuperMimicConfig().wm_observations['body_pos_b'].body_names,names),expected)

    def test_scene_names_mix_leaf_and_full_names_in_either_order(self):
        labels = ('robot/right_foot','robot/base','robot/left_foot')
        query = ('robot/left_foot','right_.*')
        self.assertEqual(resolve_scene_names(query,labels),[0,2])
        self.assertEqual(resolve_scene_names(query,labels,preserve_order=True),[2,0])
        self.assertEqual(resolve_scene_names('.*_foot',labels),[0,2])
        self.assertEqual(resolve_matching_names(('left_.*','right_.*'),('right_foot','base','left_foot'),True),([2,0],['left_foot','right_foot']))

    def test_matching_errors_include_invalid_unmatched_and_ambiguous_queries(self):
        labels = ('robot/right_foot','robot/base','robot/left_foot')
        for query,message in (((), 'at least one'),('foot','matched nothing'),('[','Invalid name pattern'),
                              (('.*_foot','robot/left_foot'),'Multiple patterns'),(('left_.*','left_foot'),'Multiple patterns')):
            with self.subTest(query=query),self.assertRaisesRegex(ValueError,message):
                resolve_scene_names(query,labels)

    def test_dictionary_matching_keeps_values_aligned_and_strict_is_optional(self):
        mapping = {'left_.*':3,'right_.*':7,'missing':9}
        names = ('right_foot','base','left_foot')
        with self.assertRaisesRegex(ValueError,'missing'):
            resolve_matching_names_values(mapping,names)
        self.assertEqual(resolve_matching_names_values(mapping,names,True,False),([2,0],['left_foot','right_foot'],[3,7]))
        self.assertEqual(resolve_matching_names_values(mapping,names,strict=False),([0,2],['right_foot','left_foot'],[7,3]))


class TermSelectionTests(unittest.TestCase):
    def test_bound_term_serializes_as_unbound_config(self):
        from newrl.utils.config import dump, load

        term = EntityField("robot", "joint_vel", joint_names=".*_knee")
        bound = term._initialize(self.scene)
        self.assertEqual(dump(bound), dump(term))
        with self.assertRaises(RuntimeError):
            _ = term.shape
        rebound = load(dump(bound))._initialize(self.scene)
        torch.testing.assert_close(
            rebound(self.physical, self.memory), bound(self.physical, self.memory)
        )

    def setUp(self):
        robot = RobotEntity(
            joint_target_indices=None, root_q_indices=None, default_position=None, action=None,
            joint_names=('robot/right_knee','robot/left_knee','robot/waist'),
            body_names=('robot/right_foot','robot/base','robot/left_foot'),
            joint_q_indices=torch.arange(3).unsqueeze(0),joint_qd_indices=torch.arange(3).unsqueeze(0),body_indices=torch.arange(3).unsqueeze(0),
            joint_mirror={'right_knee':('left_knee',1),'left_knee':('right_knee',1),'waist':('waist',1)},
        )
        model = SimpleNamespace(
            joint_limit_lower=wp.array([-1.,-2.,-10.],dtype=float,device='cpu'),
            joint_limit_upper=wp.array([1.,2.,10.],dtype=float,device='cpu'),
            body_com=wp.zeros(3,dtype=wp.vec3,device='cpu'),
        )
        self.scene = SimpleNamespace(entities={'robot':robot},model=model)
        pose = torch.zeros(2,3,7);pose[...,6]=1;pose[...,2]=torch.tensor([1.,2.,3.])
        self.physical = TensorDict({'robot':{
            'joint_pos':torch.tensor([[2.,-3.,100.],[0.,0.,100.]]),
            'joint_vel':torch.tensor([[2.,3.,100.],[4.,5.,100.]]),
            'body_pose_w':pose,'body_twist_w':torch.arange(36.).reshape(2,3,6),
        }},[2])
        self.memory = TensorDict({},[2])

    def test_joint_and_body_observations_select_and_report_shapes(self):
        observation = EntityField('robot','joint_vel',joint_names=('left_.*','right_.*'),preserve_order=True)._initialize(self.scene)
        self.assertEqual(observation.shape,(2,))
        torch.testing.assert_close(observation(self.physical,self.memory),self.physical['robot','joint_vel'][:,[1,0]])
        mirror = observation.symmetry_transform(self.scene)
        torch.testing.assert_close(mirror(torch.tensor([[3.,2.]])),torch.tensor([[2.,3.]]))
        pose = EntityField('robot','body_pose_w',body_names='.*_foot')._initialize(self.scene)
        self.assertEqual(pose.shape,(2,7))
        torch.testing.assert_close(pose(self.physical,self.memory),self.physical['robot','body_pose_w'][:,[0,2]])
        feet = BodyLinearVelocity('robot',('left_foot','right_foot'),preserve_order=True)._initialize(self.scene)
        torch.testing.assert_close(feet(self.physical,self.memory),self.physical['robot','body_twist_w'][:,[2,0],:3])
        height = BodyHeight('robot','b.*')._initialize(self.scene)
        torch.testing.assert_close(height(self.physical,self.memory),torch.full((2,1),2.))
        for term in (EntityField('robot','joint_vel',body_names='.*'),EntityField('robot','body_pose_w',joint_names='.*')):
            with self.assertRaises(ValueError):term._initialize(self.scene)

    def test_joint_reward_and_limits_ignore_unselected_joints(self):
        speed = JointVel2(.5,joint_names='.*_knee')._initialize(self.scene)
        torch.testing.assert_close(speed(self.physical,self.memory),torch.tensor([[-6.5],[-20.5]]))
        limits = JointLimit(2.,soft_limit_factor=1.,joint_names='.*_knee')._initialize(self.scene)
        torch.testing.assert_close(limits(self.physical,self.memory),torch.tensor([[-4.],[0.]]))

    def test_tracking_reward_binds_regex_once_and_retains_batch_shape(self):
        term = TrackBodyMotion(1.,.3,body_names='.*_foot')._initialize(self.scene)
        self.assertEqual(term._indices.tolist(),[0,2])
        memory = TensorDict({'reference':{'body_pose_w':self.physical['robot','body_pose_w'].clone()}},[2])
        memory['reference','body_pose_w'][:,1,2] += 100
        torch.testing.assert_close(term(self.physical,memory),torch.ones(2,1))


if __name__ == '__main__':unittest.main()
