import re
import itertools
import numpy as np
import math
from utility import *

def parse_urdf(urdf_file_path):
    '''
    TODO(Q3.1.1)

    Implement a urdf file parser and extract the origins and axes for each joint into numpy arrays.
    Arguments: string
    Returns: A dictionary with numpy arrays that contain the origins and axes of each joint
    '''
    with open('franka_robot.urdf') as fp:
        all_of_it = fp.read()
        num_joints = all_of_it.count('axis') # Change this to the number of times you encounter the word "axis" in the document
    origin = np.zeros((num_joints+1, 6))
    axis = np.zeros((num_joints+1, 3))
    joint_num = 0
    with open(urdf_file_path) as fp:
        for cnt, line in enumerate(fp):
            if (re.search("origin", line.strip())):
                ll = [e.split() for e in re.findall(r'"(.*?)"', line.strip())]
                ll.reverse()
                ori = list(itertools.chain(*ll))
                if len(ori) == 6:
                    ori = [eval(e.replace('$','').replace('{','').replace('}',''), {}, {'pi': np.pi}) for e in ori]
                    origin[joint_num] = ori
            elif (re.search("axis", line.strip())):
                axi = [e.split() for e in re.findall(r'"(.*?)"', line.strip())]
                axi = list(itertools.chain(*axi))
                if len(axi) == 3:
                    axi = [eval(e.replace('$','').replace('{','').replace('}',''), {}, {'pi': math.pi}) for e in axi]
                    axis[joint_num] = axi
                    joint_num += 1

    # Since the end-effector transformation is not included in the franka urdf, I will manually provide
    # the transformation here for you from the flange frame.
    origin[-1,2] = 0.1034
    # print(origin)
    return {'origin': origin, 'axis': axis}

class FrankaRobot():

    def __init__(self, urdf_file_path, dh_params, num_dof):

        self.robot_params = parse_urdf(urdf_file_path)
        self.dh_params = dh_params
        self.num_dof = num_dof
        print(num_dof)

    def forward_kinematics_urdf(self, joints):
        '''
        TODO(Q3.1.2)

        Calculate the position of each joint using the robot_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''
        forward_kinematics = np.zeros((self.robot_params['axis'].shape[0],4,4))
        for i in range(forward_kinematics.shape[0]):
            x, y, z, r, p, yy = self.robot_params["origin"][i]
            if i < self.num_dof:
                yy += joints[i]
            #T = translation_matrix([x, y, z]) @ rotx_matrix(r) @ roty_matrix(p) @ rotz_matrix(yy)
            T = np.dot(translation_matrix([x, y, z]), np.dot(rotx_matrix(r), np.dot(roty_matrix(p), rotz_matrix(yy))))
            if i == 0:
                forward_kinematics[i,:,:] = T
            else:
                #forward_kinematics[i,:,:] = forward_kinematics[i - 1] @ T
                forward_kinematics[i,:,:] = np.dot(forward_kinematics[i - 1] , T)
        return forward_kinematics

    def forward_kinematics_dh(self, joints):
        '''
        TODO(Q3.2.1)

        Calculate the position of each joint using the dh_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''

        forward_kinematics = np.zeros((self.dh_params.shape[0],4,4))
        for i in range(forward_kinematics.shape[0]):
            a ,d, alpha, theta = self.dh_params[i]
            if i < self.num_dof:
                theta += joints[i]
            #T = rotx_matrix(alpha) @ translation_matrix([a, 0, 0]) @ translation_matrix([0, 0, d]) @ rotz_matrix(theta)
            T = np.dot(rotx_matrix(alpha) , np.dot(translation_matrix([a, 0, 0]) , np.dot(translation_matrix([0, 0, d]) , rotz_matrix(theta))))
            # print(T)
            if i == 0:
                forward_kinematics[i] = T
            else:
                #forward_kinematics[i] = forward_kinematics[i - 1] @ T
                forward_kinematics[i,:,:] = np.dot(forward_kinematics[i - 1] , T)
        return forward_kinematics

    def ee(self, joints):
        '''
        TODO(Q3.2.2)

        Use one of your forward kinematics implementations to return the position of the end-effector.
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''
        #? what is the origin though
        Tee = self.forward_kinematics_dh(joints)[-1]
        x, y, z = trans_vec(Tee)
        R = Tee[0:3, 0:3]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        return np.array([x,y,z,roll,pitch,yaw])

#     def jacobian(self, joints):
#         '''
#         TODO(Q4.1.1)

#         Calculate the end-effector jacobian analytically using your forward kinematics
#         Arguments: array of joint positions (rad)
#         Returns: A numpy array that contains the 6 x num_dof end-effector jacobian.
#         '''

#         jacobian = np.zeros((6,self.num_dof))
#         FK = self.forward_kinematics_dh(joints)
#         for i in range(self.num_dof):
#             axis = self.robot_params["axis"][i].reshape(3,-1)
#             axis = (rot_matrix(FK[i]) @ axis).reshape(3,)
#             p = trans_vec(FK[-1]) - trans_vec(FK[i])
#             jacobian[:, i] = np.concatenate((np.cross(axis, p), axis))
#         return jacobian

#     def inverse_kinematics(self, desired_ee_pos, current_joints):
#         '''
#         TODO(Q5.1.1)

#         Implement inverse kinematics using one of the methods mentioned in class.
#         Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
#                    current_joints which represents the current location of the robot
#         Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
#         '''
#         alpha = 0.5
#         joints = np.zeros(self.num_dof)
#         current_ee_pos = self.ee(current_joints)
#         while not np.allclose(current_ee_pos, desired_ee_pos):
#             current_ee_pos = self.ee(current_joints)
#             del_x = desired_ee_pos - current_ee_pos
#             del_q = alpha * self.jacobian(current_joints).T @ del_x.reshape(-1, 1)
#             current_joints = current_joints + del_q.reshape(7,)
#         joints = np.array(current_joints)
#         return joints

#     def check_box_collision(self, joints, box):
#         '''
#         TODO(Q6.1.1)

#         Implement collision checking with a box.
#         Arguments: joints represents the current location of the robot
#                    box contains the position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]
#         Returns: A boolean where True means the box is in collision with the arm and false means that there are no collisions.
#         '''



#         return in_collision

# if __name__ == "__main__":
#     dh_params = np.array([[0, 0.333, 0, 0],
#                           [0, 0, -math.pi / 2, 0],
#                           [0, 0.316, math.pi / 2, 0],
#                           [0.0825, 0, math.pi / 2, 0],
#                           [-0.0825, 0.384, -math.pi / 2, 0],
#                           [0, 0, math.pi / 2, 0],
#                           [0.088, 0, math.pi / 2, 0],
#                           [0, 0.107, 0, 0],
#                           [0, 0.1034, 0, 0]])
#     fr = FrankaRobot('franka_robot.urdf', dh_params, 7)
#     joints = np.array([0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])
#     fr.check_box_collision(joints, [1])

if __name__ == '__main__':
    #print(parse_urdf("franka_robot.urdf"))
    dh_params = np.array([[0, 0.333, 0, 0],
                          [0, 0, -math.pi/2, 0],
                          [0, 0.316, math.pi/2, 0],
                          [0.0825, 0, math.pi/2, 0],
                          [-0.0825, 0.384, -math.pi/2, 0],
                          [0, 0, math.pi/2, 0],
                          [0.088, 0, math.pi/2, 0],
                          [0, 0.107, 0, 0],
                          [0, 0.1034, 0, 0]])
    fr = FrankaRobot('franka_robot.urdf', dh_params, 7)
    joints = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4]
    urdf_fk = fr.forward_kinematics_urdf(joints)
    #print(urdf_fk)
    dh_fk = fr.forward_kinematics_dh(joints)
    #print(dh_fk)
    
    print(np.allclose(urdf_fk, dh_fk))