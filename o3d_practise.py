import open3d as o3d
import os
import numpy as np

# pcd = o3d.io.read_point_cloud('/home/u910/Desktop/point cloud/1.ply')
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd],width=800,height=800)
# o3d.io.write_point_cloud('/home/u910/Desktop/point cloud/copy_of_1.pcd',pcd)
#
# img = o3d.io.read_image('rainbow.rgb')
# o3d.visualization.draw_geometries([img], window_name='show image',width=800,
#                                   height=800,left=50,top=50,
#                                   mesh_show_back_face=False)
# o3d.io.write_image('copy1.png',img)

import copy

# # 在原点创建坐标框架网格
# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# # 往x方向平移1.3米
# mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
# # 往y方向平移1.3米
# mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
# # 打印网格中心坐标
# print(f'Center of mesh: {mesh.get_center()}')
# print(f'Center of mesh tx: {mesh_tx.get_center()}')
# print(f'Center of mesh ty: {mesh_ty.get_center()}')
# # 可视化
# o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])

# # 在原点创建坐标框架网格
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# 使用欧拉角创建旋转矩阵
mesh_r = copy.deepcopy(mesh)
R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
# 旋转网格
# mesh_r.rotate(R, center=(0, 0, 0))
#
# mesh_r.scale(0.5, ceter=mesh_r.get_center())
# 可视化
# o3d.visualization.draw_geometries([mesh, mesh_r])

#通用变换
T = np.eye(4)
T[:3,:3] = mesh.get_rotation_matrix_from_xyz((np.pi/2,np.pi/4,np.pi/3))
T[0,3] = 1
T[1,3] = 1.3
mesh_t = copy.deepcopy(mesh).transform(T)
o3d.visualization.draw_geometries([mesh,mesh_t])
