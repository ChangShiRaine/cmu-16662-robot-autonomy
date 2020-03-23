import math
import numpy as np
import xml.etree.ElementTree as ET
from tf.transformations import quaternion_matrix
from pyquaternion import Quaternion

def translation_matrix(trans_vec):
    trans= np.array(
        [[1,0,0,trans_vec[0]],
        [0,1,0,trans_vec[1]],
        [0,0,1,trans_vec[2]],
        [0,0,0,1]]
        )

    return trans



def rotx_matrix(roll):
    rotx= np.array(
        [[1,0,0,0],
        [0,np.cos(roll),-np.sin(roll),0],
        [0,np.sin(roll),np.cos(roll),0],
        [0,0,0,1]]
        )

    return rotx

def roty_matrix(pitch):
    roty= np.array(
        [[np.cos(pitch),0,np.sin(pitch),0],
        [0,1,0,0],
        [-np.sin(pitch),0,np.cos(pitch),0],
        [0,0,0,1]]
        )

    return roty

def rotz_matrix(yaw):
    rotz= np.array(
        [[np.cos(yaw), -np.sin(yaw),0,0],
        [np.sin(yaw), np.cos(yaw),0,0],
        [0,0,1,0],
        [0,0,0,1]]
        )

    return rotz

def parse_urdf(urdf_file_path):
    '''
    TODO(Q3.1.1)
    Implement a urdf file parser and extract the origins and axes for each joint into numpy arrays.
    Arguments: string
    Returns: A dictionary with numpy arrays that contain the origins and axes of each joint 
    '''
    pi = np.pi
    tree = ET.parse('franka_robot.urdf')
    root = tree.getroot()
    joints = root[0].findall('joint')

    num_joints = len(joints) # Change this to the number of times you encounter the word "axis" in the document
    origin = np.zeros((num_joints+1, 6))
    axis = np.zeros((num_joints+1, 3))

    # Since the end-effector transformation is not included in the franka urdf, I will manually provide
    # the transformation here for you from the flange frame.
    origin[-1,2] = 0.1034

    for i in range(num_joints):
    	joint = joints[i]
    	origin_rpy = joint.findall('origin')[0].attrib['rpy'].replace('$','').replace('{','').replace('}','').split(' ')
    	origin_xyz = joint.findall('origin')[0].attrib['xyz'].split(' ')
    	origin_value = origin_xyz + origin_rpy
    	axis_value = joint.findall('axis')[0].attrib['xyz'].split(' ')

    	for j in range(len(origin_value)):
    		origin[i, j] = eval(origin_value[j])

    	for j in range(len(axis_value)):
    		axis[i, j] = eval(axis_value[j])

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
        forward_kinematics = np.zeros((self.robot_params['axis'].shape[0],4,4))
 
        for i in range(forward_kinematics.shape[0]):
            x, y, z, roll, pitch, yaw = self.robot_params['origin'][i]
 
            if i < len(joints):
                yaw += joints[i]

            T = np.dot(translation_matrix([x, y, z]), np.dot(rotx_matrix(roll), np.dot(roty_matrix(pitch), rotz_matrix(yaw))))
 
            if i == 0:
                forward_kinematics[i] = T
            else:
                forward_kinematics[i] = np.dot(forward_kinematics[i-1] , T)

        #print(forward_kinematics)
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
            a, d, alpha, theta= self.dh_params[i]      
            if i < len(joints):
                theta += joints[i]  

            T = np.dot(np.dot(np.dot(rotx_matrix(alpha), translation_matrix([a,0,0])), translation_matrix([0,0,d])), rotz_matrix(theta))   

            if i == 0:
                forward_kinematics[i] = T
            else:
                forward_kinematics[i] = np.dot(forward_kinematics[i-1], T)

        #print(forward_kinematics)
        return forward_kinematics

    def ee(self, joints):
        '''
        TODO(Q3.2.2)
        Use one of your forward kinematics implementations to return the position of the end-effector.
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''

        T = self.forward_kinematics_urdf(joints)[-1]
        x, y, z = T[0:3,-1]
        R = T[0:3,0:3]

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular :
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else :
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0

        return np.array([x,y,z,roll,pitch,yaw])

    def jacobian(self, joints):
        '''
        TODO(Q4.1.1)
        Calculate the end-effector jacobian analytically using your forward kinematics
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 6 x num_dof end-effector jacobian.
        '''
        jacobian = np.zeros((6,self.num_dof))
        forward_kinematics = self.forward_kinematics_urdf(joints)

        for i in range(self.num_dof):
            axis = self.robot_params["axis"][i].reshape(3,1)
            axis = np.dot(forward_kinematics[i][0:3,0:3], axis).reshape(3,)
            p = (forward_kinematics[-1][0:3,-1] - forward_kinematics[i][0:3,-1]).reshape(3,)
            jacobian[:, i] = np.concatenate((np.cross(axis, p), axis))

        return jacobian

    def inverse_kinematics(self, desired_ee_pos, current_joints):
        '''
        TODO(Q5.1.1)
        Implement inverse kinematics using one of the methods mentioned in class. 
        Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
                   current_joints which represents the current location of the robot
        Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
        '''
        alpha = 0.1
        current_ee_pos = self.ee(current_joints)
        while not np.linalg.norm(current_ee_pos - desired_ee_pos) <  0.0001:
        #while not np.allclose(current_ee_pos, desired_ee_pos):
            delta_x = desired_ee_pos - current_ee_pos
            delta_q = alpha * np.dot(self.jacobian(current_joints).T, delta_x.reshape(-1, 1))
            current_joints += delta_q.reshape(self.num_dof,)
            current_ee_pos = self.ee(current_joints)
        joints = np.array(current_joints)
        return joints

    def check_box_collision(self, joints, box):
        '''
        TODO(Q6.1.1)
        Implement collision checking with a box.
        Arguments: joints represents the current location of the robot
                   box contains the position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]

        Returns: A boolean where True means the box is in collision with the arm and false means that there are no collisions.
        '''
        in_collision = False

        # translation(x,y,z), quaternion(w,x,y,z), size(l,w,h) for 12 boxes
        trans_vecs = np.array(
            [[-0.04, 0, -0.283],
            [-0.009, 0, -0.183],
            [-0, -0.032, -0.082],
            [-0.008, 0, 0],
            [-0, 0.042, 0.067],
            [0.00687, 0, -0.139],
            [-0.008, 0.004, 0],
            [0.00422, 0.05367, -0.121],
            [0.00422, 0.00367, -0.263],
            [-0.00328, 0.0176, -0.0055],
            [-0.00136, 0.0092, 0.0083],
            [-0.00136, 0.0092, 0.1407]])

        quaternions = np.array(
            [[1, 0, 0, 0],
            [1, 0, 0, 0],
            [0.9514, 0.3079, 0, 0],
            [1, 0, 0, 0],
            [0.9514, 0.3079, 0, 0],
            [1, 0, 0, 0],
            [0.7071, -0.7071, 0, 0],
            [0.9962, -0.08715, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0.9239, 0, 0, -0.3827]])

        sizes = np.array(
            [[0.23, 0.2, 0.1],
            [0.13, 0.12, 0.1],
            [0.12, 0.1, 0.2],
            [0.15, 0.27, 0.11],
            [0.12, 0.1, 0.2],
            [0.13, 0.12, 0.25],
            [0.13, 0.23, 0.15],
            [0.12, 0.12, 0.4],
            [0.12, 0.12, 0.25],
            [0.13, 0.23, 0.12],
            [0.12, 0.12, 0.2],
            [0.08, 0.22, 0.17]])

        # calculate the pos of the obstacle box object
        #x, y, z, roll, pitch, yaw, obj_W, obj_H, obj_D = box
        x, y, z, roll, pitch, yaw, obj_D, obj_W, obj_H = box

        obj_W /= 2 
        obj_H /= 2 
        obj_D /= 2

        obj_pos = np.dot(translation_matrix([x, y, z]), np.dot(rotx_matrix(roll), np.dot(roty_matrix(pitch), rotz_matrix(yaw))))
        obj_center = obj_pos[0:3,-1]
        obj_axis_x = obj_pos[0:3, 0]
        obj_axis_y = obj_pos[0:3, 1]
        obj_axis_z = obj_pos[0:3, 2]

        forward_kinematics = self.forward_kinematics_dh(joints)

        # get arm box location
        centers = np.zeros((12, 3))
        axis_xs = np.zeros((12, 3))
        axis_ys = np.zeros((12, 3))
        axis_zs = np.zeros((12, 3))

        for i in range(12):
            R = quaternion_matrix(quaternions[i,:])
            trans = translation_matrix(trans_vecs[i,:])
            # print("rot=", np.dot(trans,R))

            #R = np.identity(4)
            #quaternion = Quaternion(w=quaternions[i,0], x=quaternions[i,1], y=quaternions[i,2], z=quaternions[i,3])
            #R[0:3, 0:3] = quaternion.rotation_matrix

            # assignt the translation matrix to transformation matrix
            R[0:3, 3] = trans_vecs[i,:].T
            if i <= 4:
                box_pos = np.dot(np.dot(forward_kinematics[0], trans), R)
                #box_pos = np.dot(forward_kinematics[0], R)
            if i == 5:
                box_pos = np.dot(np.dot(forward_kinematics[2], trans), R)
                #box_pos = np.dot(forward_kinematics[2], R)
            if i == 6:
                box_pos = np.dot(np.dot(forward_kinematics[3], trans), R)
                #box_pos = np.dot(forward_kinematics[3], R)
            if i >= 7 and i <= 9:
                box_pos = np.dot(np.dot(forward_kinematics[4], trans), R)
                #box_pos = np.dot(forward_kinematics[4], R)
            if i == 10 or i == 11:
                box_pos = np.dot(np.dot(forward_kinematics[6], trans), R)
                #box_pos = np.dot(forward_kinematics[6], R)

            centers[i,:] = box_pos[0:3,-1]
            axis_xs[i,:] = box_pos[0:3, 0] 
            axis_ys[i,:] = box_pos[0:3, 1]
            axis_zs[i,:] = box_pos[0:3, 2]

        single_box_collision = np.zeros(12)

        for i in range(12):
            collision = 1
            T = obj_center - centers[i]
            Ls = np.zeros((15,3))

            #CASE 1: L = Ax
            Ls[0,:] = axis_xs[i]

            #CASE 2: L = Ay
            Ls[1,:] = axis_ys[i]

            #CASE 3: L = Az
            Ls[2,:] = axis_zs[i]

            #CASE 4: L = Bx
            Ls[3,:] = obj_axis_x

            #CASE 5: L = By
            Ls[4,:] = obj_axis_y

            #CASE 6: L = Bz
            Ls[5,:] = obj_axis_z

            #CASE 7: L = Ax cross Bx
            Ls[6,:] = np.cross(axis_xs[i], obj_axis_x)

            #CASE 8: L = Ax cross By
            Ls[7,:] = np.cross(axis_xs[i], obj_axis_y)

            #CASE 9: L = Ax cross Bz
            Ls[8,:] = np.cross(axis_xs[i], obj_axis_z)

            #CASE 10: L = Ay cross  Bx
            Ls[9,:] = np.cross(axis_ys[i], obj_axis_x)

            #CASE 11: L = Ay cross By
            Ls[10,:] = np.cross(axis_ys[i], obj_axis_y)

            #CASE 12: L = Ay cross Bz
            Ls[11,:] = np.cross(axis_ys[i], obj_axis_z)

            #CASE 13: L = Az cross Bx
            Ls[12,:] = np.cross(axis_zs[i], obj_axis_x)

            #CASE 14: L = Az cross By
            Ls[13,:] = np.cross(axis_zs[i], obj_axis_y)

            #CASE 15: L = Az cross Bz
            Ls[14,:] = np.cross(axis_zs[i], obj_axis_z)

            DA = sizes[i,0]/2 
            WA = sizes[i,1]/2 
            HA = sizes[i,2]/2 

            for j in range(15):
                L = Ls[j,:]
                center_proj = np.abs(np.dot(T, L))
                edge_proj = np.abs(WA * np.dot(axis_xs[i], L)) + \
                    np.abs(HA *np.dot(axis_ys[i], L)) + \
                    np.abs(DA *np.dot(axis_zs[i], L)) + \
                    np.abs(obj_W*np.dot(obj_axis_x, L)) + \
                    np.abs(obj_H*np.dot(obj_axis_y, L)) + \
                    np.abs(obj_D*np.dot(obj_axis_z, L))
                
                if center_proj > edge_proj:
                    collision = 0
                    break

            single_box_collision[i] = collision

        if np.sum(single_box_collision) > 0:
            in_collision = True

        return in_collision

if __name__== '__main__':
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
    box = [0.5, 0, 0.21, 0, 0, 0, 0.15, 0.09, 0.126]
    print(fr.check_box_collision(joints, box))