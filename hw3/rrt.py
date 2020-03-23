from time import time
import numpy as np

from kdtree import KDTree
from franka_robot import FrankaRobot


class SimpleTree:

    def __init__(self, dim):
        self._parents_map = {}
        self._kd = KDTree(dim)

    def insert_new_node(self, point, parent=None):
        node_id = self._kd.insert(point)
        self._parents_map[node_id] = parent

        return node_id
        
    def get_parent(self, child_id):
        return self._parents_map[child_id]

    def get_point(self, node_id):
        return self._kd.get_node(node_id).point

    def get_nearest_node(self, point):
        return self._kd.find_nearest_point(point)


class RRT:

    def __init__(self, fr, is_in_collision):
        self._fr = fr
        self._is_in_collision = is_in_collision

        '''
        TODO: You can tune these parameters to improve RRT performance.

        However, make sure the values satisfy the following conditions:
            self._constraint_th < 2e-3
            self._q_step_size < 0.1
        '''
        self._project_step_size = 1e-1
        self._constraint_th = 1.9e-3

        self._q_step_size = 0.09
        self._target_p = 0.3
        self._max_n_nodes = int(1e5)


    def sample_valid_joints(self):
        '''
        TODO: Implement sampling a random valid configuration.

        The sampled configuration must be within the joint limits, but it does not check for collisions.

        Please use the following in your code:
            self._fr.joint_limits_low - lower joint limits
            self._fr.joint_limits_high - higher joint limits
            self._fr.num_dof - the degree of freedom of franka
        '''

        q = np.random.random_sample((self._fr.num_dof,)) * (self._fr.joint_limits_high - self._fr.joint_limits_low) + self._fr.joint_limits_low

        return q

    def project_to_constraint(self, q, constraint):
        '''
        TODO: Implement projecting a configuration to satisfy a constraint function using gradient descent.

        Please use the following parameters in your code:
            self._project_step_size - learning rate for gradient descent
            self._constraint_th - a threshold lower than which the constraint is considered to be satisfied

        Input:
            q - the point to be projected
            constraint - a function of q that returns (constraint_value, constraint_gradient)
                         constraint_value is a scalar - it is 0 when the constraint is satisfied
                         constraint_gradient is a vector of length 6 - it is the gradient of the
                                constraint value w.r.t. the end-effector pose (x, y, z, r, p, y)

        Output:
            q_proj - the projected point

        You can obtain the Jacobian by calling self._fr.jacobian(q)
        '''
        q_proj = q.copy()
        err, grad = constraint(q_proj)
        while err > self._constraint_th:
            #print("err", err)
            q_proj -= self._project_step_size * np.dot(self._fr.jacobian(q_proj).T, np.array(grad))
            err, grad = constraint(q_proj)
        return q_proj

    def extend(self, tree, q_target, constraint=None):
        '''
        TODO: Implement the constraint extend function.

        Input: 
            tree - a SimpleTree object containing existing nodes
            q_target - configuration of the target state
            constraint - a constraint function used by project_to_constraint
                         do not perform projection if constraint is None

        Output:
            target_reached - bool, whether or not the target has been reached
            new_node_id - node_id of the new node inserted into the tree by this extend operation
                         Note: tree.insert_new_node returns a node_id
        '''
        while True:
            if np.random.random() < self._target_p:
                q_sample = q_target
            else:
                q_sample = self.sample_valid_joints()

            nearest_node_id, dist = tree.get_nearest_node(q_sample)
            q_near = tree.get_point(nearest_node_id)

            extend_direction_norm = np.linalg.norm(q_sample - q_near)
            q_new = q_near + np.minimum(self._q_step_size, extend_direction_norm) * (q_sample - q_near) / extend_direction_norm
            if constraint != None:
                q_new = self.project_to_constraint(q_new, constraint)
            if self._is_in_collision(q_new):
                continue
            new_node_id = tree.insert_new_node(q_new, nearest_node_id)
            if np.linalg.norm(q_new - q_target) < self._q_step_size:
                target_reached = True
            else:
                target_reached = False
            return target_reached, new_node_id

    def plan(self, q_start, q_target, constraint=None):
        tree = SimpleTree(len(q_start))
        tree.insert_new_node(q_start)

        s = time()
        for n_nodes_sampled in range(self._max_n_nodes):
            if n_nodes_sampled > 0 and n_nodes_sampled % 100 == 0:
                print('RRT: Sampled {} nodes'.format(n_nodes_sampled))

            #print("Extending...")
            reached_target, node_id_new = self.extend(tree, q_target, constraint)

            if reached_target:
                break

        print('RRT: Sampled {} nodes in {:.2f}s'.format(n_nodes_sampled, time() - s))

        path = []
        if reached_target:
            backward_path = [q_target]
            node_id = node_id_new
            while node_id is not None:
                #print("node_id", node_id)
                backward_path.append(tree.get_point(node_id))
                node_id = tree.get_parent(node_id)
            path = backward_path[::-1]

            print('RRT: Found a path! Path length is {}.'.format(len(path)))
        else:
            print('RRT: Was not able to find a path!')
        
        return path

if __name__ == '__main__':
    np.random.seed(0)
    rospy.init_node('rrt')
    fr = FrankaRobot()

    boxes = np.array([
        # obstacle
        [0.4, 0, 0.25, 0, 0, 0, 0.3, 0.05, 0.5],
        # sides
        [0.15, 0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
        [0.15, -0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
        # back
        [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
        # front
        [0.75, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
        # top
        [0.2, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
        # bottom
        [0.2, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01]
    ])

    def is_in_collision(joints):
        for box in boxes:
            if fr.check_box_collision(joints, box):
                return True
        return False

    desired_ee_rp = fr.ee(fr.home_joints)[3:5]
    def ee_upright_constraint(q):
        '''
        TODO: Implement constraint function and its gradient. 
        
        This constraint should enforce the end-effector stays upright.
        Hint: Use the roll and pitch angle in desired_ee_rp. The end-effector is upright in its home state.

        Input:
            q - a joint configuration

        Output:
            err - a non-negative scalar that is 0 when the constraint is satisfied
            grad - a vector of length 6, where the ith element is the derivative of err w.r.t. the ith element of ee
        '''
        ee = fr.ee(q)

        err, grad = None, None
        return err, grad

    joints_start = fr.home_joints.copy()
    joints_start[0] = -np.deg2rad(45)
    joints_target = joints_start.copy()
    joints_target[0] = np.deg2rad(45)

    rrt = RRT(fr, is_in_collision)

    print(rrt.sample_valid_joints())
