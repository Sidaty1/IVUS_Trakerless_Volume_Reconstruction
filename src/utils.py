
from scipy.spatial.transform import Rotation
from numpy import asarray






def euler_to_matrix(euler):
    rot = Rotation.from_euler("xyz", euler)
    return rot

def matrix_to_euler(rot):
    euler = rot.as_euler("xyz", degrees=True)
    return asarray(euler)

def matrix_to_rot_and_trans(mat):
    rot = [
        [mat[0][0], mat[0][1], mat[0][2]], 
        [mat[1][0], mat[1][1], mat[1][2]],
        [mat[2][0], mat[2][1], mat[2][2]]
    ]
    rot = Rotation.from_matrix(rot)
    return rot, asarray([mat[0][3], mat[1][3], mat[2][3]])

def get_start_point(l):
    return [int(a.split('.')[0]) for a in l].sort()[0]