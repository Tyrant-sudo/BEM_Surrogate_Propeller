import numpy as np
from . import util
from .bernstein  import bernstein_poly, trivariate_bernstein
import csv,ast
import matplotlib.pyplot as plt
from mayavi import mlab

csv.field_size_limit(500 * 1024 * 1024)
def xyz_to_stu(xyz, origin, stu_axes):
    if stu_axes.shape == (3,):
        stu_axes = np.diag(stu_axes)
        # raise ValueError(
        #     'stu_axes should have shape (3,), got %s' % str(stu_axes.shape))
    # s, t, u = np.diag(stu_axes)
    assert(stu_axes.shape == (3, 3))
    s, t, u = stu_axes
    tu = np.cross(t, u)
    su = np.cross(s, u)
    st = np.cross(s, t)

    diff = xyz - origin

    # TODO: vectorize? np.dot(diff, [tu, su, st]) / ...
    stu = np.stack([
        np.dot(diff, tu) / np.dot(s, tu),
        np.dot(diff, su) / np.dot(t, su),
        np.dot(diff, st) / np.dot(u, st)
    ], axis=-1)
    return stu


def stu_to_xyz(stu_points, stu_origin, stu_axes):
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points*stu_axes


def get_stu_control_points(dims):
    stu_lattice = util.mesh3d(
        *(np.linspace(0, 1, d+1) for d in dims), dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_points


def get_control_points(dims, stu_origin, stu_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, stu_origin, stu_axes)
    return xyz_points


def get_stu_deformation_matrix(stu, dims):
    v = util.mesh3d(
        *(np.arange(0, d+1, dtype=np.int32) for d in dims),
        dtype=np.int32)

    v = np.reshape(v, (-1, 3))

    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32),
        v=v,
        stu=np.expand_dims(stu, axis=-2))
    
    b = np.prod(weights, axis=-1)
    return b


def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    

    # for i in stu:
    #     print(i[1])
    return get_stu_deformation_matrix(stu, dims)


def get_ffd(xyz, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(xyz)
    
    b = get_deformation_matrix(xyz, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(xyz):
    minimum, maximum = util.extent(xyz, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes

if __name__ == '__main__':
    header1 = ('name','coordinate')

    test_mesh = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\mesh_data.csv'
    mesh_plot = []
    with open(test_mesh,'r') as f:
        reader = csv.DictReader(f)
        lenth = 0
        for i in reader:
            lenth+=1
            print(lenth)
            mesh_plot.append(ast.literal_eval(i[header1[1]]))
            if lenth>10:
                break
    mesh_plot1 = np.array(mesh_plot)[0].reshape(-1,3)
    mesh_plot2 = np.array(mesh_plot)[1].reshape(-1,3)
    dim = np.array([3,4,2])
    # dim = 3
    b,p = get_ffd(mesh_plot1,dim)

    print(b.shape,p.shape)
    # print(b[5],p[59])
    b2,p2 = get_ffd(mesh_plot2,dim)
    # print(p,p2)
    # p = np.dot(b,p2)
    p = p2
    # exit()
    meshx = mesh_plot1[:,0]
    meshy = mesh_plot1[:,1]
    meshz = mesh_plot1[:,2]
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]

    
    # s = mlab.points3d(meshx,meshy,meshz,scale_factor=.005)
    s = mlab.points3d(x,y,z,scale_factor=.005)
    mlab.show()