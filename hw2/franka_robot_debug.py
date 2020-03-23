# -*- coding: utf-8 -*-

import math
import numpy as np
import roslib; roslib.load_manifest('urdfdom_py')
import rospy
import xml.etree.ElementTree as ET
# from scipy.spatial.transform import Rotation as R
#from pyquaternion import Quaternion
from tf.transformations import quaternion_matrix

def parse_urdf(urdf_file_path):
    num_joints = 0 # Change this to the number of times you encounter the word "axis" in the document
    joint_axis_oris = {}
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    orgin = np.array

    for axis in root.iter('axis'):
        num_joints += 1

    # print('The total number of axis in the document is', num_joints)

    origin = np.zeros((num_joints+1, 6))
    axis = np.zeros((num_joints+1, 3))
    iter_num = 0

    for axis_xml in root.iter('axis'):
        iter_num += 1
        current_axis_num = axis_xml.attrib['xyz'].split()
        for i in range(0,3):
            axis[iter_num,i] = float(current_axis_num[i])
    
    iter_num = 0
    for origin_xml in root.iter('origin'):
        if iter_num == 0:
            iter_num += 1
        else:
            iter_num += 1
            rpy = origin_xml.attrib['rpy'].split()
            xyz = origin_xml.attrib['xyz'].split()
            for i in range(3):
                origin[iter_num-2, i] = float(xyz[i])
                try:
                    origin[iter_num-2, i+3] = float(rpy[i])
                except:
                    if rpy[i][2] == "p":
                        origin[iter_num-2, i+3] = math.pi/2
                        # print(origin)
                    else:
                        origin[iter_num-2, i+3] = -math.pi/2
                
    '''
    TODO(Q3.1.1)
    
    Implement a urdf file parser and extract the origins and axes for each joint into numpy arrays.
    Arguments: string
    Returns: A dictionary with numpy arrays that contain the origins and axes of each joint 
    '''

    

    # Since the end-effector transformation is not included in the franka urdf, I will manually provide
    # the transformation here for you from the flange frame.
    # temp_swap_origin = origin[:, 0:3]
    # origin[:,0:3] = origin[:,4:5]
    # origin[:,3:] = temp_swap_origin
    # origin[:,] = 
    origin[-1,2] = 0.1034
    # print(origin)

    return {'origin': origin, 'axis': axis}

class FrankaRobot():

    def __init__(self, urdf_file_path, dh_params, num_dof):

        self.robot_params = parse_urdf(urdf_file_path)
        self.dh_params = dh_params
        self.num_dof = num_dof

    def forward_kinematics_urdf(self, joints):
        '''
        TODO(Q3.1.2)
        
        Calculate the position of each joint using the robot_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''
        axis = self.robot_params['axis']
        origin = self.robot_params['origin']
        temp_joint = joints
        temp_joint.append(0)
        temp_joint.append(0)


        forward_kinematics = np.zeros((self.robot_params['axis'].shape[0],4,4))

        prev_transform = np.identity(4)

        for i in range(axis.shape[0]):
            # z(pitch) is the 3th element of axis
            # y(yaw) is the 2th element of axis
            # x(row) is the 1th element of axis
            ro_z = origin[i,5]
            z_rotationmatrix = np.array([[math.cos(ro_z), -math.sin(ro_z), 0, 0],
                                         [math.sin(ro_z), math.cos(ro_z), 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])
            ro_y = origin[i,4]
            y_rotationmatrix = np.array([[math.cos(ro_y), 0, math.sin(ro_y), 0],
                                         [0, 1, 0, 0],
                                         [-math.sin(ro_y), 0, math.cos(ro_y), 0],
                                         [0, 0, 0, 1]])
            ro_x = origin[i,3]
            x_rotationmatrix = np.array([[1, 0, 0, 0],
                                         [0, math.cos(ro_x), -math.sin(ro_x), 0],
                                         [0, math.sin(ro_x), math.cos(ro_x), 0],
                                         [0, 0, 0, 1]])

            #x, y, z is the 4, 5, 6 element of axis
            trans_x = origin[i,0]
            trans_y = origin[i,1]
            trans_z = origin[i,2]
            translation = np.array([[1,0,0,trans_x],
                                    [0,1,0,trans_y],
                                    [0,0,1,trans_z],
                                    [0,0,0,1]])

            rotation = np.dot(x_rotationmatrix,np.dot(y_rotationmatrix,z_rotationmatrix))
            current_transform = np.dot(translation, rotation)
            local_transform = np.dot(prev_transform, current_transform)

            # if i < (len(joints)):
            #     print(i)
            joint_rotation = np.array([[math.cos(temp_joint[i]), -math.sin(temp_joint[i]), 0, 0],
                                       [math.sin(temp_joint[i]), math.cos(joints[i]), 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
            total_transform = np.dot(local_transform, joint_rotation)

            forward_kinematics[i, :, :] = total_transform

            prev_transform = total_transform

        return forward_kinematics

    def forward_kinematics_dh(self, joints):
        '''
        TODO(Q3.2.1)
        
        Calculate the position of each joint using the dh_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''

        forward_kinematics = np.zeros((self.dh_params.shape[0],4,4))
        prev_transform = np.identity(4)

        for i in range(self.dh_params.shape[0]):
            a = self.dh_params[i,0]
            d = self.dh_params[i,1]
            alpha = self.dh_params[i,2]
            theta = joints[i] if (i < 7) else (0)
            rot_z = np.array([[math.cos(theta), -math.sin(theta), 0, 0],
                              [math.sin(theta), math.cos(theta), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            rot_x = np.array([[1, 0, 0, 0],
                             [0, math.cos(alpha), -math.sin(alpha), 0],
                             [0, math.sin(alpha), math.cos(alpha), 0],
                             [0, 0, 0, 1]])
            trans_z = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, d],
                                [0, 0, 0, 1]])
            trans_x = np.array([[1, 0, 0, a],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            current_transform = np.dot(rot_x,np.dot(trans_x,np.dot(trans_z,rot_z)))
            total_transform = np.dot(prev_transform, current_transform)

            forward_kinematics[i, :, :] = total_transform

            prev_transform = total_transform

        return forward_kinematics

    def ee(self, joints):
        '''
        TODO(Q3.2.2)
        
        Use one of your forward kinematics implementations to return the position of the end-effector.
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''
        forward_kinematics = self.forward_kinematics_dh(joints)
        # print(forward_kinematics)
        x = forward_kinematics[-1,0,3]
        # =print(x)
        y = forward_kinematics[-1,1,3]
        # print(y)
        z = forward_kinematics[-1,2,3]
        # print(z)

        R = forward_kinematics[-1,0:3,0:3]
        

        phi = 0.0
        if np.allclose(R[2,0],-1.0):
            theta = math.pi/2.0
            psi = math.atan2(R[0,1],R[0,2])
        elif np.allclose(R[2,0],1.0):
            theta = -math.pi/2.0
            psi = math.atan2(-R[0,1],-R[0,2])
        else:
            theta = -math.asin(R[2,0])
            cos_theta = math.cos(theta)
            psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
            phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)

        if (forward_kinematics[-1][1][1] < 1e-3):
            roll = math.pi/2
        if (forward_kinematics[-1][0][1] < 1e-3):
            yaw = math.pi/2

        roll = psi
        pitch = theta
        yaw = phi

        return np.array([x, y, z, roll, pitch, yaw]) # it should be 6 element array


    def jacobian(self, joints):
        '''
        TODO(Q4.1.1)
        
        Calculate the end-effector jacobian analytically using your forward kinematics
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 6 x num_dof end-effector jacobian.
        '''
        # I call the function here, and print x, y, z seperately in the function
        # return x, y ,z in a 6 element array.....and the return is weird
        a = self.ee(joints)
        # print(a)

        forward_kinematics = self.forward_kinematics_dh(joints)
        endEffectorPos = forward_kinematics[-1, 0:3, 3]
        # print(endEffectorPos)

        # below is not relevant I didn't print anything aftere that

        # print(forward_kinematics)
        jacobian = np.zeros((6,self.num_dof))
        for i in range(self.num_dof):
            rotAxis = forward_kinematics[i, 0:3, 2]
            # print(rotAxis)
            jointCPos = forward_kinematics[i, 0:3, 3]
            # print(jointCPos)
            jacobian[:3,i] = np.cross(rotAxis, endEffectorPos - jointCPos)
            jacobian[3:,i] = rotAxis

        return jacobian

    def inverse_kinematics(self, desired_ee_pos, current_joints):
        '''
        TODO(Q5.1.1)
        
        Implement inverse kinematics using one of the methods mentioned in class. 
        Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
                   current_joints which represents the current location of the robot
        Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
        '''
        joints = np.zeros(self.num_dof)
        current_ee_pos = self.ee(current_joints)
        delta_x = desired_ee_pos - current_ee_pos
        # print('delta_x', delta_x)
        lr = 0.6

        i = 0
        
        # if np.sum(delta_x) > 0.0000001 :
        while (np.sqrt(np.sum(np.square(delta_x)))> 1e-4):
            
            i+=1
            # print(1)

            current_ee_pos = self.ee(current_joints)
            # print("current_ee_pos = ", current_ee_pos.shape)
            delta_x = desired_ee_pos - current_ee_pos

            # if i%1000 == 0:
                #print(np.sqrt(np.sum(np.square(delta_x))))
                # print(np.sqrt(np.sum(np.square(delta_x))))


            J = self.jacobian(current_joints)
            # print("J.shape=", J.shape)
            delta_q = lr * np.dot(J.transpose(), delta_x)
            # print("delta_q.shape=", delta_q.shape)
            current_joints += delta_q
            # print("current_j1eoints.shape=",current_joints.shape)
            # print(desired_ee_pos-current_ee_pos)

        # print('inverse_return_length=',len(current_joints))
        return current_joints[:7]

    def check_box_collision(self, joints, box):
        '''
        TODO(Q6.1.1)
        
        Implement collision checking with a box.
        Arguments: joints represents the current location of the robot
                   box co1entains the position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]
        Returns: A boolean where True means the box is in collision with the arm and false means that there are no collisions.
        '''



        """1. get the unit vector of the axis of each box"""
            # 1.1 Calculate the forward kinematic for the joints, and get rotation and translation
            # 1.2 Dot the translation and rotation from the table
            # 1.3 Get the unit vector

        # 2. get the "cross product" unit vector of the axis of each box
        # 3. look through all the "unit vector" to check the collision

        BoxesConfig = [{"Link":1, "Refer":1, "Translation":np.array([-0.04,0,-0.283]),   "Rotation":np.array([1,0,0,0]), "Size": np.array([0.23,0.2,0.1])},
                       {"Link":2, "Refer":1, "Translation":np.array([-0.009,0,-0.183]),  "Rotation":np.array([1,0,0,0]), "Size": np.array([0.13,0.12,0.1])},
                       {"Link":3, "Refer":1, "Translation":np.array([-0,-0.032,-0.082]), "Rotation":np.array([0.9514,0.3079,0,0]), "Size": np.array([0.12,0.1,0.2])},
                       {"Link":4, "Refer":1, "Translation":np.array([-0.008,0,0]), "Rotation":np.array([1,0,0,0]), "Size": np.array([0.15,0.27,0.11])},
                       {"Link":5, "Refer":1, "Translation":np.array([-0,0.042,0.067]), "Rotation":np.array([0.9514,0.3079,0,0]), "Size": np.array([0.12,0.1,0.2])},
                       {"Link":6, "Refer":3, "Translation":np.array([0.00687,0,-0.139]), "Rotation":np.array([1,0,0,0]), "Size": np.array([0.13,0.12,0.25])},
                       {"Link":7, "Refer":4, "Translation":np.array([-0.008,0.004,0]),   "Rotation":np.array([0.7071,-0.7071,0,0]), "Size": np.array([0.13,0.23,0.15])},
                       {"Link":8, "Refer":5, "Translation":np.array([0.00422,0.05367,-0.121]), "Rotation":np.array([0.9962,-0.08715,0,0]), "Size": np.array([0.12,0.12,0.4])},
                       {"Link":9, "Refer":5, "Translation":np.array([0.00422,0.00367,-0.263]), "Rotation":np.array([1,0,0,0]), "Size": np.array([0.12,0.12,0.25])},
                       {"Link":10, "Refer":5, "Translation":np.array([-0.00328,0.0176,-0.0055]), "Rotation":np.array([1,0,0,0]), "Size": np.array([0.13,0.23,0.12])},
                       {"Link":11, "Refer":7, "Translation":np.array([-0.00136,0.0092,0.0083]), "Rotation":np.array([0,1,0,0]), "Size": np.array([0.12,0.12,0.2])},
                       {"Link":12, "Refer":7, "Translation":np.array([-0.00136,0.0092,0.1407]), "Rotation":np.array([0.9239,0,0,-0.3827]), "Size": np.array([0.08,0.22,0.17])}]

        # (joints_number, 4, 4)
        B_roll = box[3]
        B_pitch = box[4]
        B_yaw = box[5]

        z_rotationmatrix = np.array([[math.cos(B_yaw), -math.sin(B_yaw), 0, 0],
                                     [math.sin(B_yaw), math.cos(B_yaw), 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
        y_rotationmatrix = np.array([[math.cos(B_pitch), 0, math.sin(B_pitch), 0],
                                     [0, 1, 0, 0],
                                     [-math.sin(B_pitch), 0, math.cos(B_pitch), 0],
                                     [0, 0, 0, 1]])
        x_rotationmatrix = np.array([[1, 0, 0, 0],
                                     [0, math.cos(B_roll), -math.sin(B_roll), 0],
                                     [0, math.sin(B_roll), math.cos(B_roll), 0],
                                     [0, 0, 0, 1]])

        box_rot_matrix = np.dot(z_rotationmatrix,np.dot(y_rotationmatrix,x_rotationmatrix))[0:3,0:3]

        # box_rot_matrix = R.from_euler('xyz',np.array([B_roll, B_yaw, B_pitch], degrees=False))
        Bx = box_rot_matrix [0:3, 0]

        By = box_rot_matrix [0:3, 1]
    
        Bz = box_rot_matrix [0:3, 2]

        PB = np.array([box[0],box[1],box[2]]).transpose() #(3,1)
        DB = box[6]/2
        WB = box[7]/2
        HB = box[8]/2

        forward_kinematics = self.forward_kinematics_dh(joints)

        box_num = len(BoxesConfig)

        boxes_transformation = np.ones((box_num, 4, 4))        

        all_box_collision = np.zeros((box_num,1))
        all_box_collision[:,:] = True
        left_haha = np.zeros((box_num,15))
        right_haha = np.zeros((box_num,15))


        for i, box in enumerate(BoxesConfig):
            joint_to_box = np.ones((4,4))

            world_to_joint = np.identity(4)
            world_to_joint[:,:] = forward_kinematics[box["Refer"]-1, :, :] # (4,4)
            # if i == 0:
            #     print("world_to_joint ", world_to_joint)

            # assign the rotation matrix to transformation matrix
            box_rot = box["Rotation"]
            #quaternion = Quaternion(w=box_rot[0], x=box_rot[1], y=box_rot[2], z=box_rot[3])
            # r = R.from_quat(box["Rotation"])
            joint_to_box = np.identity(4)
            joint_to_box[0:4, 0:4] = quaternion_matrix(box_rot)

            # assignt the translation matrix to transformation matrix
            joint_to_box[0:3, 3] = box["Translation"].transpose()
            # if i == 0:
            #     print(joint_to_box)

            world_to_box = np.dot(world_to_joint, joint_to_box)
            # if i == 0:
            #     print("world_to_box ", world_to_box)
            
            boxes_transformation[i, :, :] = world_to_box

            Ax = world_to_box [0:3, 0]
            # print(Ax)
            Ay = world_to_box [0:3, 1]
            Az = world_to_box [0:3, 2]

            AxBx = np.cross(Ax, Bx)
            AxBy = np.cross(Ax, By)
            AxBz = np.cross(Ax, Bz)

            AyBx = np.cross(Ay, Bx)
            AyBy = np.cross(Ay, By)
            AyBz = np.cross(Ay, Bz)

            AzBx = np.cross(Az, Bx)
            AzBy = np.cross(Az, By)
            AzBz = np.cross(Az, Bz)

            DA = box["Size"][0]/2
            WA = box["Size"][1]/2
            HA = box["Size"][2]/2

            PA = world_to_box[0:3,3]
            T = PB - PA

            all_case = [Ax, Ay, Az, Bx, By, Bz, AxBx, AxBy, AxBz, AyBx, AyBy, AyBz, AzBx, AzBy, AzBz]


            # all_box_collision[i] = True #1
            for j,L in enumerate(all_case):
                left = np.absolute(np.dot(T, L))
                right = np.absolute(WA*np.dot(Ax,L))+\
                        np.absolute(HA*np.dot(Ay,L))+\
                        np.absolute(DA*np.dot(Az,L))+\
                        np.absolute(WB*np.dot(Bx,L))+\
                        np.absolute(HB*np.dot(By,L))+\
                        np.absolute(DB*np.dot(Bz,L))
                left_haha[i,j] = left
                right_haha[i,j] = right
                # print("left=",left)
                # print("right=",right)
                if (left > right):
                    # print("the ", i, "is collision" )
                    all_box_collision[i] = False #0 # no collision
                    break

        if np.sum(all_box_collision) == 0:
            # print(all_box_collision)
            return False
        else:
            return True
        # return all_box_collision, left_haha, right_haha


if __name__ == '__main__':
    dh_params = np.array([[0, 0.333, 0, 0],
                          [0, 0, -math.pi/2, 0],
                          [0, 0.316, math.pi/2, 0],
                          [0.0825, 0, math.pi/2, 0],
                          [-0.0825, 0.384, -math.pi/2, 0],
                          [0, 0, math.pi/2, 0],
                          [0.088, 0, math.pi/2, 0],
                          [0, 0.107, 0, 0],
                          [0, 0.1034, 0, 0]])
    # parse_urdf(path)
    myrobot = FrankaRobot('franka_robot.urdf', dh_params, 7)
    # print(myrobot.forward_kinematics_urdf(np.ones((3,3))))
    joints = np.array([0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4])
    box = [0.5, 0, 0.21, 0, 0, 0, 0.15, 0.09, 0.126]
    haha = myrobot.check_box_collision(joints,box)
    print(haha)
