#!/usr/bin/env python3
import sys
import os
import copy
import math
import shutil
import glob
# amend relative import
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # repo path
sys.path.append(Path(__file__).resolve().parent.as_posix())  # file path
from params import *

try:
    _egg_file = sorted(Path(CARLA_PATH, 'PythonAPI/carla/dist').expanduser().glob('carla-*%d.*-%s.egg' % (
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    )))[0].as_posix()
    sys.path.append(_egg_file)
except IndexError:
    print('CARLA Egg File Not Found.')
    exit()

import carla
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.calibration import Calibration

VIEW_WIDTH = 1242
VIEW_HEIGHT = 375
VIEW_FOV = 90
camera_intrinsic_matrix = np.identity(3)
camera_intrinsic_matrix[0, 2] = VIEW_WIDTH / 2.0
camera_intrinsic_matrix[1, 2] = VIEW_HEIGHT / 2.0
camera_intrinsic_matrix[0, 0] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
camera_intrinsic_matrix[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

#输入：type_id,id,等， frame_id.
#返回raw_lidar_data的路径
def get_ego_lidar_data_path(ego_vehicle_label, _frame_id):
    #type_id_id
    ego_vehicle_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
    #record/vehicle.tesla.model3_344
    ego_raw_data_path = raw_data_path / ego_vehicle_name
    #sensor.lidar.ray_cast_344
    #iterdir在pathlib中，相当于os.listdir(p) == p.iterdir()
    #三个文件夹，rgb, label, ply
    raw_data_file = sorted(ego_raw_data_path.iterdir())
    lidar_raw_data_file = (ego_raw_data_path / raw_data_file[-1] / (_frame_id + 'ply')).as_posix()
    # ego_raw_data_path = raw_data_path + '/' + ego_vehicle_name
    # raw_data_file = os.listdir(ego_raw_data_path)
    # raw_data_file.sort()
    # lidar_raw_data_file = ego_raw_data_path + '/' + raw_data_file[-1] + '/' + _frame_id + 'ply'
    return lidar_raw_data_file

#返回ego_vehicle文件夹中label文件夹下对应的_frame_id的文件
def get_ego_camera_2Dlabel_path(ego_vehicle_label, camera_id, _frame_id):
    ego_vehicle_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
    ego_raw_data_path = raw_data_path / ego_vehicle_name
    raw_data_file = sorted(ego_raw_data_path.iterdir())
    # ego_raw_data_path = raw_data_path + '/' + ego_vehicle_name
    # raw_data_file = os.listdir(ego_raw_data_path)
    # raw_data_file.sort()
    for _file in raw_data_file:
        if str(camera_id) in str(_file) and 'label' in str(_file):
            camera_2Dlabel_file = (ego_raw_data_path / _file / (_frame_id + 'txt')).as_posix()
            # camera_2Dlabel_file = ego_raw_data_path + '/' + _file + '/' + _frame_id + 'txt'
            return camera_2Dlabel_file

#将点云数据转换为3D可视化矩阵
def get_raw_lidar_data(lidar_raw_data_file, sensor_rotation):
    pcd = o3d.io.read_point_cloud(lidar_raw_data_file)
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi, 0, np.pi / 2]))
    tmp_R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation)
    # pcd.rotate(R, center=(0,0,0))
    # pcd.rotate(tmp_R,center=(0,0,0))
    pointcloud = np.array(pcd.points, dtype=np.dtype('f4')) #np.dtype(float32)
    #右手系换左手系
    pointcloud = np.dot(pointcloud, np.array([1, 0, 0, 0, -1, 0, 0, 0, 1]).reshape(3, 3))
    #将nparray的点云数据转换为可以用o3d处理的格式
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd

#返回去除衰减率的，可以被o3d处理的点云数据,将点云数据映射到二位像素中
def process_pcd(pcd, calib):
    img_shape = [1242, 375]
    pointcloud = np.array(pcd.points, dtype=np.dtype('f4'))
    #去掉x,y,z，衰减率中的衰减率, 然后将点云数据
    pts_rect = calib.lidar_to_rect(pointcloud[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(pointcloud[fov_flag])
    tmp_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return tmp_pcd


def get_fov_flag(pts_rect, img_shape, calib):
    '''
    Valid point should be in the image (and in the PC_AREA_SCOPE)
    :param pts_rect:
    :param img_shape:
    :return:
    '''
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[0])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[1])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    pts_valid_flag = np.logical_and(pts_valid_flag, pts_rect_depth <= 110)
    return pts_valid_flag

#返回vehicle_id(vehicle.tesla.model3_399)相关sensor的信息
#如果是lidar: 返回三个值：[x,y,z], [0,0,yaw], [roll,pitch,0]
#如果是camera：返回一个list， 每个元素为[[x,y,z],[0,0,yaw], sensor_id]
def get_sensor_transform(vehicle_id, labels, sensor=None):
    if sensor == 'lidar':
        for label in labels:
            if vehicle_id == label[-1] and 'lidar' in label[0]:
                #map(function, list)将list中的元素应用到function组成一个新的list
                sensor_center = np.array(list(map(float, label[2:5]))) #sensor loacation: x,y,z
                sensor_center[1] *= -1
                # sensor_rotation = np.array([-np.radians(float(label[6])),-np.radians(float(label[5])),np.radians(float(label[7]))])
                # np.radian(180)=pi 将角度转换为弧度
                sensor_rotation = np.array([0, 0, np.radians(float(label[7]))]) #[0,0,yaw]
                sensor_rotation_test = np.array([np.radians(float(test)) for test in label[5:7]] + [0])#[5:7] roll, pitch [roll,pitch,0]
                #[x,y,z], [0,0,yaw], [roll,pitch,0]
                return sensor_center, sensor_rotation, sensor_rotation_test
    elif sensor == 'camera':
        camera_info_list = []
        for label in labels:
            if vehicle_id == label[-1] and 'camera.rgb' in label[0]:
                sensor_center = np.array(list(map(float, label[2:5]))) #sensor location:[x,y,z]
                sensor_center[1] *= -1
                # sensor_rotation = np.array([-np.radians(float(label[6])),-np.radians(float(label[5])),np.radians(float(label[7]))])
                sensor_rotation = np.array([0, 0, np.radians(float(label[7]))]) #[0,0,yaw]
                camera_info_list.append([sensor_center, sensor_rotation, label[1]])
        return camera_info_list

#获取label文件中车的x,y,z, yaw,roll,pith(弧度）
def get_vehicle_transform(vehilce_label):
    location = np.array(list(map(float, vehilce_label[2:5]))) #x,y,z
    rotation = np.array(
        [np.radians(float(vehilce_label[5])), np.radians(float(vehilce_label[6])), np.radians(float(vehilce_label[7]))]) #yaw, roll,pitch
    return location, rotation

def get_traffic_light_transform(traffic_label):
    location = np.array(list(map(float, traffic_label[2:5])))
    rotation = np.array(
        [np.radians(float(traffic_label[5])), np.radians(float(traffic_label[6])), np.radians(float(traffic_label[7]))]
    )
    return location, rotation

#通过other_vehicle的label获取他们的世界坐标系，转换到摄像头坐标系，
#返回bbox(min_x,min_y,max_x,max_y), bbox_extend(extend.x,y,z),bbox_center(x,y,z) bbox_delta:other_vehicle和selsor的夹角弧度
#如果flag为true返回一个扩大后的bbox_extend
def get_vehicle_bbox(other_vehilcle_label, sensor_center, sensor_rotation, flag=False):
    #location, 注意这个location中的z,加了汽车的高度，同时y变为-y
    bbox_center = np.array(list(map(float, other_vehilcle_label[2:5]))) + np.array(
        [0, 0, float(other_vehilcle_label[-1])])
    #坐标系转换（左右手？）
    bbox_center[1] *= -1
    #获得sensor和车之间的旋转平移矩阵
    sensor_world_matrix = get_matrix(sensor_center, -sensor_rotation + [0, 0, np.pi * 4 / 2])
    # sensor_world_matrix = get_matrix(sensor_center,-sensor_rotation+[0,0,np.pi*3/2])
    #取逆world到sensor的转换矩阵
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    #将box_center的世界坐标系转为camera坐标系，是4*4 4*1两个矩阵相乘
    bbox_center = np.append(bbox_center, 1).reshape(4, 1)
    bbox_center = np.dot(world_sensor_matrix, bbox_center).tolist()[:3]
    #将other_vehicle的yaw转换为弧度，算出和sensor的相对弧度，sensor和other_vehicle的夹角
    tmp = sensor_rotation - [0, 0, np.radians(float(other_vehilcle_label[7]))]
    # tmp = [0,0,sensor_rotation[2]-np.radians(float(other_vehilcle_label[7]))]
    # print(sensor_rotation,[np.radians(float(test)) for test in other_vehilcle_label[5:8]])
    # tmp_tmp = [-sensor_rotation[1]+np.radians(float(other
    # _vehilcle_label[6])),-sensor_rotation[0]+np.radians(float(other_vehilcle_label[5]))]+[0]
    #获得sensor和other_vehicle夹角的旋转矩阵。 get_rotation_matrix_from_xyz, 根据输入的(pitch,roll,yaw）弧度，算出相应的旋转矩阵
    bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tmp)
    # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation-[0,0,np.radians(float(other_vehilcle_label[7]))])
    #yaw
    bbox_delta = sensor_rotation[2] - np.radians(float(other_vehilcle_label[7])) - np.pi / 2
    # print(bbox_delta)
    #bbox_extend: extend.x, y , z
    bbox_extend = np.array([float(num) * 2 for num in other_vehilcle_label[-4:-1]])
    if flag:
        bbox_extend += np.array([0.1, 0.1, 0.05])
    #输入：bb_center的世界坐标location，sensor和other_vehicle的夹角弧度，bbox的extend
    #输出：对应摄像头中bbox的参数
    bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
    bbox.color = np.array([0.5, 1.0, 0.5])
    return bbox, bbox_extend, bbox_center, bbox_delta

#输入：record, frame, ego_vehicle所有信息那行，camera_index(都是0）, calib的list
#输出：把calib_info这个list存入calib00文件夹下，同时返回一个calibration类，这个类中包含了calib的信息
def get_calib_file(raw_data_path, frame_id, ego_vehicle_label, index, calib_info):
    ego_vehicle_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
    #dataset/ record / vehicle.tesla.model3_233
    dataset_path = (COOK_DATA_PATH / raw_data_path.stem / ego_vehicle_name).as_posix()
    # dataset_path = 'dataset/' + raw_data_path[4:] + '/' + ego_vehicle_name
    if not os.path.exists(dataset_path + '/calib0' + str(index)):
        os.makedirs(dataset_path + '/calib0' + str(index))
    np.savetxt(dataset_path + '/calib0' + str(index) + '/' + frame_id, np.array(calib_info), fmt='%s', delimiter=' ')
    calib = Calibration(dataset_path + '/calib0' + str(index) + '/' + frame_id)
    return calib

#raw_data_path: raw_data/record, _frame, ego_vehicle在label中的信息， tmp_bbox:所有other_vehicle的extend,location和sensor夹角信息，
#pointCloud:将点云源文件中的数据转换为o3d可以处理的类型, index:0, camera_id, calib_infor_list:校准矩阵
def write_labels(raw_data_path, frame_id, ego_vehicle_label, tmp_bboxes, pointcloud, index, camera_id, calib_info):
    #vehicle.tesla.model3_344
    ego_vehicle_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
    #dataset/record/vehicle.tesla.model3_344
    dataset_path = (COOK_DATA_PATH / raw_data_path.stem / ego_vehicle_name).as_posix()
    # dataset_path = 'dataset/' + raw_data_path[4:] + '/' + ego_vehicle_name
    #raw_data/record/vehicle.tesla.model3_344/
    _raw_data_path = (raw_data_path / ego_vehicle_name).as_posix()
    # raw_data_path += '/' + ego_vehicle_name
    sensor_raw_path = os.listdir(_raw_data_path)
    #label00
    if not os.path.exists(dataset_path + '/label0' + str(index)):
        os.makedirs(dataset_path + '/label0' + str(index))
    #image00
    if not os.path.exists(dataset_path + '/image0' + str(index)):
        os.makedirs(dataset_path + '/image0' + str(index))
    #calib00
    if not os.path.exists(dataset_path + '/calib0' + str(index)):
        os.makedirs(dataset_path + '/calib0' + str(index))
    #velodyne00
    if not os.path.exists(dataset_path + '/velodyne'):
        os.makedirs(dataset_path + '/velodyne')
    #[raw_data/record/vechile下的三个文件夹]
    #将raw_data中的图像转存到datasets/vehicle中
    for _tmp in sensor_raw_path:
        if str(camera_id) in _tmp and 'label' not in _tmp:
            image_path = Path(_raw_data_path, _tmp, frame_id[:-3] + 'png').as_posix()
            # image_path = raw_data_path + '/' + _tmp + '/' + frame_id[:-3] + 'png'
            break
    shutil.copy(image_path, dataset_path + '/image0' + str(index))
    # np.savetxt(dataset_path + '/calib0'+str(index)+'/' + frame_id, np.array(calib_info), fmt='%s', delimiter=' ')

    #通过calib文件获得内参，外参的转换矩阵
    calib = Calibration(dataset_path + '/calib0' + str(index) + '/' + frame_id)

    #将数据存放到dataset/vehicle/label00中
    for idx, tmp_bbox in enumerate(tmp_bboxes):
        coordinate = np.array(list(map(float, tmp_bbox[11:14]))).reshape(1, 3)
        coordinate_camera = calib.lidar_to_rect(coordinate).flatten()
        tmp_bboxes[idx][11:14] = coordinate_camera
    np.savetxt(dataset_path + '/label0' + str(index) + '/' + frame_id, np.array(tmp_bboxes), fmt='%s', delimiter=' ')

    #将点云数据存放到velodyne文件夹中，格式为.bin
    pointcloud = np.array(pointcloud.points, dtype=np.dtype('f4'))
    # pointcloud[:,2] = pointcloud[:,2] * -1
    # points_r = np.zeros((pointcloud.shape[0],1),dtype=np.dtype('f4'))
    points_R = np.exp(-0.05 * np.sqrt(np.sum(pointcloud ** 2, axis=1))).reshape(-1, 1)
    pointcloud = np.concatenate((pointcloud, points_R), axis=1)
    #numpy: tofile将数组中的数据以2进制的格式存入
    pointcloud.tofile(dataset_path + '/velodyne/' + frame_id[:-3] + 'bin')
    return image_path

#根据rotation和location获得旋转和平移矩阵
def get_matrix(location, rotation):
    T = np.matrix(np.eye(4))
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
    T[0:3, 3] = location.reshape(3, 1)
    return T

#输入：lidar center, lidar rotation, camera_sensor, camera_rotation
#获得lidar坐标系到camera坐标系的旋转平移矩阵
def get_matrix_from_origin_to_target(origin_location, origin_orientation, target_location, target_orientation):

    origin_matrix = get_matrix(origin_location, origin_orientation)
    target_matrix = get_matrix(target_location, target_orientation)
    #np.linalg.inv返回输入矩阵的逆矩阵
    target_matrix = np.linalg.inv(target_matrix)
    T = np.dot(target_matrix, origin_matrix)
    return T
    # delta_rotation = origin_orientation - target_orientation
    # delta_translate = origin_location - target_location
    # T = np.eye(4)
    # T[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(delta_rotation)
    # T[0:3,3] = delta_translate
    # print(T)
    # return T


def transform_vehicle_to_lidar(vehicle_location, matrix):
    old_location = np.append(vehicle_location, 1).reshape((4, 1))
    new_location = np.dot(matrix, old_location).flatten()[:3]
    return new_location


def transform_lidar_to_camera(vehicle_bbox_points, intrinsic_matrix, extrinsic_matrix):
    bb_cords = np.transpose(np.concatenate((np.array(vehicle_bbox_points), np.ones([8, 1])), axis=1))
    cords_x_y_z = np.dot(extrinsic_matrix, bb_cords)[:3, :]
    cords_x_minus_z_y = np.concatenate([cords_x_y_z[0, :], -cords_x_y_z[2, :], cords_x_y_z[1, :]])
    bbox = np.transpose(np.dot(intrinsic_matrix, cords_x_minus_z_y))
    camera_bbox = np.transpose(np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]]))
    return camera_bbox


def filter_label(label):
    label = label.reshape(3, 8)
    x_min, y_min = np.min(label, axis=1)[:2]
    x_max, y_max = np.max(label, axis=1)[:2]
    depth_min = np.min(label, axis=1)[2]
    # print(depth_min)
    # print(y_min,y_max)
    # y_min -= 40*depth_min/100
    # y_max -= 30*depth_min/100
    # print(y_min,y_max)
    return [x_min, y_min, x_max, y_max]

#将相机内参矩阵3*3转换为标准的3*4矩阵
def post(camera_intrinsic_matrix, index):
    info = []
    info.append('P' + str(index) + ':')
    for row in camera_intrinsic_matrix:
        for column in row:
            info.append(str(column))
        info.append(str(0))
    return info

#输入：lidar坐标系到camera坐标系的一个转换矩阵
#输出：一个3*4内部转换矩阵
def process_matrix(velo_to_cam_matrix):
    test = np.identity(4)
    test =  np.concatenate([-test[1, :], -test[2, :], test[0, :], test[3,:]]).reshape(4,4)
    velo_to_cam_matrix = np.dot(velo_to_cam_matrix, test)
    info = []
    info.append('Tr_velo_to_cam:')
    for row in velo_to_cam_matrix.tolist():
        for column in row:
            info.append(str(column))
            if len(info) == 13:
                return info

#input:vehicles:[vehicle.tesla.model3_232, vehicle.tesla.model3_233]
#labels: type_id, id, location, rotation, boundingbox的四个值 extend.x,y,z, boundingbox.location.z
#output: 返回一个list, key: type_id_id, value:返回所有AD_vehicle对应摄像头的位置，旋转角度，以及camera_id
def find_Ad_vehicles_location(vehicles, labels):
    results = {}
    # return {}
    # print(labels)
    #vehicle.tesla.model3_455
    for vehicle in vehicles:
        lidar_location, lidar_rotation, tmp_rotation = get_sensor_transform(vehicle[-3:], labels, 'lidar')
        camera_location, camera_rotation, id = get_sensor_transform(vehicle[-3:], labels, 'camera')[0]
        results[vehicle] = [lidar_location, lidar_rotation, tmp_rotation, camera_location, camera_rotation, id]
    return results

#traffic_lights:[traffic.traffic_light_91,]
#labels： type_id, id, location, rotation, [0,0,0] id
def find_traffic_light_location(traffic_lights, labels):
    results = {}
    for traffic_light in traffic_lights:
        lidar_location, lidar_rotation, tmp_rotation = get_sensor_transform(traffic_light[-2:],labels,'lidar')
        camera_location, camera_rotation,id = get_sensor_transform(traffic_light[-2:],labels,'camera')[0]
        results[traffic_light] = [lidar_location,lidar_rotation,tmp_rotation,camera_location,camera_rotation,id]
    return results



def judge_in_ROI_numbers(other_vehilcle_label, AD_vehicles_location, _frame_id):
    ans = 0

    for AD_vehicle, location in AD_vehicles_location.items():
        if other_vehilcle_label[1] in AD_vehicle:
            # print(other_vehilcle_label,AD_vehicle,'-----------------------')
            ans += 1
            continue
        AD_vehicle_path = raw_data_path + '/' + AD_vehicle
        raw_data_file = os.listdir(AD_vehicle_path)
        raw_data_file.sort()
        lidar_raw_data_file = AD_vehicle_path + '/' + raw_data_file[-1] + '/' + _frame_id + 'ply'
        pcd = get_raw_lidar_data(lidar_raw_data_file, location[2])
        ego_raw_data_path = raw_data_path + '/' + AD_vehicle
        raw_data_file = os.listdir(ego_raw_data_path)
        raw_data_file.sort()
        for _file in raw_data_file:
            if str(location[5]) in _file and 'label' in _file:
                camera_2Dlabel_file = ego_raw_data_path + '/' + _file + '/' + _frame_id + 'txt'
                break
        try:
            camera_2Dlabel = np.loadtxt(camera_2Dlabel_file).reshape(-1, 5)
        except:
            camera_2Dlabel = np.array([])
        other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label, location[0],
                                                                              location[1], True)
        tmp2 = pcd.crop(other_bbox)
        tmp_2Dlabel = []
        for _label in camera_2Dlabel:
            if other_vehilcle_label[1] == str(int(_label[0])):
                tmp_2Dlabel = _label[1:]
                if tmp_2Dlabel[0] < 0:
                    tmp_2Dlabel[0] = 0
                if tmp_2Dlabel[1] < 0:
                    tmp_2Dlabel[1] = 0
                if tmp_2Dlabel[2] > 1242:
                    tmp_2Dlabel[2] = 1242
                if tmp_2Dlabel[3] > 375:
                    tmp_2Dlabel[3] = 375
                break
        if len(tmp2.points) >= 10 and len(tmp_2Dlabel) != 0:
            ans += 1
        else:
            pass
    return ans


if __name__ == "__main__":
    # raw_data_path = 'tmp/record' + ' '.join(['2020',str(sys.argv[1][:4]),str(sys.argv[1][-4:])])
    # .resolve:将相对路径变为绝对路径
    # raw_data/record_2021_0701_1006
    raw_data_path = Path(sys.argv[1]).resolve()
    # raw_data/record/label
    label_file_path = (raw_data_path / 'label').as_posix()
    # 返回文件夹中文件名字组成的列表
    frames = os.listdir(label_file_path)


    def _read_imageset_file(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [int(line) for line in lines]


    # gt_split_file = 'dataset/record2020_0903_0142_/img_list_test.txt'
    '''# frames = _read_imageset_file(gt_split_file)
    #COOK_DATA_PATH: dataset /record2021_0624_2149 / global_label 存放所有actor(vehicle,lidar,camera,traffic_light)的信息
    #stem只有文件名，suffix只有后缀, name全名
    '''
    #dataset/record/global_label
    global_label_file_path = COOK_DATA_PATH / raw_data_path.stem / 'global_label'
    global_label_file_path.mkdir(parents=True, exist_ok=True)
    global_label_file_path = global_label_file_path.as_posix()
    # global_label_file_path = 'dataset/' + raw_data_path[4:] + '/global_label/'
    frames.sort()

    if len(frames) < 10:
        print("there is no enough data.")
    # exit()
    print('the len of collected frames', len(frames))

    frame_start = RAW_DATA_START #60
    frame_end = RAW_DATA_END #-10
    frame_hz = RAW_DATA_FREQ #1 (Hz)
    # frame_start,frame_end,frame_hz = 60,-10,1
    for _frame in frames[frame_start:frame_end:frame_hz]: #[60:-10:1]
        # continue
        # if '683' not in _frame:
        #     continue
        # _frame = str(_frame).rjust(10,'0') + '.txt'
        #将raw_data下面label文件夹60帧到最后10帧的数据拷贝到dataset文件夹下的global_label下面
        if not os.path.exists(global_label_file_path):
            os.makedirs(global_label_file_path)
        #label_file_path: raw_data/record/labeldx
        _frame_label_file_path = Path(label_file_path, _frame).as_posix()
        #shutil.copy(source,destination)
        shutil.copy(_frame_label_file_path, global_label_file_path)
        #labels: 每一帧world中所有actor的属性数据
        labels = np.loadtxt(_frame_label_file_path, dtype='str', delimiter=' ')
        # tmp_car_id = ['282','274','284','275','276']
        # AD_vehicles = [v for v in raw_data_path.iterdir() if 'vehicle' in v]
        #AD_vehicles是type_id_id文件夹名字组成list [vehicle.tesla.model3_232, vehicle.tesla.model3_233]
        AD_vehicles = [v for v in os.listdir(str(raw_data_path)) if 'vehicle' in v] #raw_data/record_2021_0701_1006
        #[traffic.traffic_light_91,]
        Traffic_lights = [tf for tf in os.listdir(str(raw_data_path)) if 'traffic' in tf]
        #返回所有AD_vehicle对应摄像头的位置，旋转角度，以及camera_id 的一个dict: key为vehicle.tesla.model3_233, value:一个包含这些信息的list
        # input:vehicles:[vehicle.tesla.model3_232, vehicle.tesla.model3_233]
        # labels: type_id, id, location, rotation, boundingbox的四个值 extend.x,y,z, boundingbox.location.z
        # output: 返回一个list, key: type_id_id, value:返回所有AD_vehicle对应摄像头的位置，旋转角度，以及camera_id
        AD_vehicles_location = find_Ad_vehicles_location(AD_vehicles, labels)
        traffic_lights_location = find_traffic_light_location(Traffic_lights,labels)

        #ego_vehicle_label: label文件中每一行的数据
        for ego_vehicle_label in labels:
            #vehicle.tesla.model3_233
            tmp_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
            # tmp_car = ['282','274','284','275','276']
            # tmp_car = ['235','236','237','238','240','275']
            if 'vehicle.tesla' in ego_vehicle_label[
                0]:  # and tmp_name in AD_vehicles_location.keys():# and '319' in ego_vehicle_label[1]:
                # if ego_vehicle_label[1] not in tmp_car:
                #     continue
                try:
                    lidar_center, lidar_rotation, lidar_rotation_test = get_sensor_transform(ego_vehicle_label[1],
                                                                                             labels, sensor='lidar')
                except:
                    continue
                # camera_center, camera_rotation = get_sensor_transform(ego_vehicle_label[1], labels, sensor='camera')
                #记录个中坐标系转换信息，最终存储在dataset/record/vechicle.tesla.model3_444/calib00下
                calib_info_list = []
                for index in range(4):
                    #将post中的list用‘ ‘连成一个新的字符串， P0: a 0 u 0 0 b v 0 0 0 1 0 (3*4)  a,b是旋转，u,v是平移
                    calib_info_list.append(' '.join(post(camera_intrinsic_matrix, index)))
                #R0_rect是lidar坐标系和camera坐标系的转换 3*3
                tmp_info = ['R0_rect:'] + list(map(str, [1, 0, 0, 0, 1, 0, 0, 0, 1]))
                calib_info_list.append(' '.join((copy.deepcopy(tmp_info))))
                #这个应该是imu和点云坐标系转换，可以不用管
                tmp_info[0] = 'Tr_imu_to_velo:'
                calib_info_list.append(' '.join(tmp_info))
                calib_info_list.append(' '.join(tmp_info))
                #返回一个只有一个元素的list, 这个元素还是list. vehicle对应sensor:camera的信息. 如果是camera：返回一个list， 每个元素为[[x,y,z],[0,0,yaw], sensor_id]
                camera_info_list = get_sensor_transform(ego_vehicle_label[1], labels, sensor='camera')
                #_frame[:-3]: frame_id.
                #返回点云数据文件路径
                lidar_raw_data_file = get_ego_lidar_data_path(ego_vehicle_label, _frame[:-3])
                #lidar_rotate_test:[roll, pitch,0]
                #输出：将点云源文件中的数据转换为o3d可以处理的类型
                pointCloud = get_raw_lidar_data(lidar_raw_data_file, lidar_rotation_test)
                #如果是camera：返回一个list， 每个元素为[[x,y,z],[0,0,yaw], sensor_id]
                #index只有一个0，
                for index, camera in enumerate(camera_info_list):
                    camera_center, camera_rotation, camera_id = camera
                    #返回从lidar坐标系到camera坐标系的转换矩阵4*4
                    velo_to_cam_matrix = get_matrix_from_origin_to_target(lidar_center, lidar_rotation, camera_center,
                                                                       camera_rotation)
                    # 返回一个3*4的内部转换矩阵
                    post_velo_to_cam = process_matrix(velo_to_cam_matrix)
                    calib_info_list[-2] = ' '.join(post_velo_to_cam)
                    #输入：raw_data/record, frame, ego_vehicle所有信息那行，camera_index(0), calib的list
                    ##输出：把calib_info这个list存入calib00文件夹下，同时返回一个calibration类，这个类中包含了calib的信息
                    calib = get_calib_file(raw_data_path, _frame, ego_vehicle_label, index, calib_info_list)
                    #去除原始数据中的衰减率，返回可以被o3d处理的点云数据
                    tmp_pointCloud = process_pcd(pointCloud, calib)
                    ##获取label文件中车的x,y,z, yaw,roll,pith(弧度）
                    ego_vehicle_location, ego_vehicle_rotation = get_vehicle_transform(ego_vehicle_label)
                    tmp_labels, bboxes = [], []
                    #存放other_vehicles的bbox数据
                    label2d = []
                    label2point = [[], []]
                    #other_vehicles中的点云数据
                    tmp_p = []
                    # 返回ego_vehicle文件夹中label文件夹下对应的_frame_id的文件, 里面有摄像头范围内other_vehicle的id, min_x,min_y,max_x,max_y(就是相机图像坐标系的东西）
                    #bbox包含的内容就是min_x,min_y,max_x,max_y
                    camera_2Dlabel_file = get_ego_camera_2Dlabel_path(ego_vehicle_label, camera_id, _frame[:-3])
                    try:
                        camera_2Dlabel = np.loadtxt(camera_2Dlabel_file).reshape(-1, 5)
                    except:
                        camera_2Dlabel = np.array([])
                    for other_vehilcle_label in labels:
                        #先筛选出record/label中是车的actor, 再筛选出不是ego_vehicle的other_vehicle
                        if 'vehicle' in other_vehilcle_label[0]:
                            if ego_vehicle_label[1] == other_vehilcle_label[
                                1]:  # or ego_vehicle_label[1] != '247' or other_vehilcle_label[1] not in ['229','216']:
                                continue
                            # object_detected_number = judge_in_ROI_numbers(other_vehilcle_label,AD_vehicles_location,_frame[:-3])
                            object_detected_number = 0
                            # exit()
                            other_location, other_rotation = get_vehicle_transform(other_vehilcle_label)

                            # tmp获取的bbox_extend比tmp2获取的bbox_extend小一些
                            other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,
                                                                                                  lidar_center,
                                                                                                  lidar_rotation)
                            tmp = tmp_pointCloud.crop(other_bbox)
                            other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,
                                                                                                  lidar_center,
                                                                                                  lidar_rotation, True)
                            tmp2 = tmp_pointCloud.crop(other_bbox)
                            # if len(tmp.points) != len(tmp2.points):
                            # if True:
                            #     print(len(tmp.points),len(tmp2.points),ego_vehicle_label[1],other_vehilcle_label[1],_frame)
                            #存放从raw_data/record/vehicle/label中读取的对应other_vehicle的bbox信息
                            tmp_2Dlabel = []
                            #camera_2Dlabel: 在record/vehicle/label文件夹下的bbox信息， 包括other_vehicle_id, min_x,min_y,max_x,max_y
                            for _label in camera_2Dlabel:
                                if other_vehilcle_label[1] == str(int(_label[0])):
                                    tmp_2Dlabel = _label[1:]
                                    if tmp_2Dlabel[0] < 0:
                                        tmp_2Dlabel[0] = 0
                                    if tmp_2Dlabel[1] < 0:
                                        tmp_2Dlabel[1] = 0
                                    if tmp_2Dlabel[2] > 1242:
                                        tmp_2Dlabel[2] = 1242
                                    if tmp_2Dlabel[3] > 375:
                                        tmp_2Dlabel[3] = 375
                                    break
                            #tmp是那个bbox类
                            if len(tmp.points) >= 10 and len(tmp_2Dlabel) != 0:
                                # if True:
                                # camera_label = transform_lidar_to_camera(other_bbox.get_box_points(),camera_intrinsic_matrix, velo_to_cam_matrix)
                                pcd = o3d.geometry.PointCloud()
                                #获取bbox中的点云数据，格式为nparray
                                tmp_pcd = np.array(np.array(other_bbox.get_box_points()), dtype=np.dtype('f4'))
                                #Vector3dVector将nparray的数据转换为open3d type
                                pcd.points = o3d.utility.Vector3dVector(tmp_pcd)
                                pcd.paint_uniform_color([1, 0, 0])
                                #tmp_p存放other_vehicles的point_cloud数据
                                tmp_p.append(pcd)

                                # if not all(camera_label.reshape(8,3)[:, 2] > 0):
                                #     print('# filter objects behind camera')
                                #     continue
                                # camera_label = filter_label(camera_label)
                                # if camera_label[0] > 1242 or camera_label[1] > 375 or camera_label[2] < 0 or camera_label[3] < 0:
                                #     print('# filter objects behind camera')
                                #     continue
                                #存放other_vehicles的bbox(min_x,min_y,max_x,max_y）数据
                                label2d.append(tmp_2Dlabel)
                                # camera_label = [np.array(i)[0][0] for i in camera_label]
                                other_location = other_bbox.get_center()
                                bboxes.append(other_bbox)
                                # print(other_location,float(other_vehilcle_label[-1]))
                                #location.z进行校准
                                other_location -= np.array([0, 0, float(other_vehilcle_label[-1])])
                                # other_location = transform_vehicle_to_lidar(other_location, matrix_to_ego)
                                #
                                alpha = - bbox_delta - math.atan(other_location[0] / other_location[2]) - np.pi


                                def process_theta(alpha):
                                    while alpha > np.pi:
                                        alpha -= np.pi * 2
                                    while alpha < -np.pi:
                                        alpha += np.pi * 2
                                    return alpha


                                alpha = process_theta(alpha)
                                # print(-bbox_delta-np.pi+lidar_rotation[2]+np.pi/2,np.radians(float(other_vehilcle_label[7])))
                                bbox_delta = process_theta(-bbox_delta - np.pi)
                                if len(tmp2.points) > 10:
                                    size = 0
                                    if len(tmp2.points) + other_location[0] < 250:
                                        size = 1
                                    if len(tmp2.points) + other_location[0] < 125:
                                        size = 2
                                    tmp_labels.append(
                                        ['Car', str(object_detected_number), str(size), str(alpha), tmp_2Dlabel[0],
                                         tmp_2Dlabel[1], tmp_2Dlabel[2], tmp_2Dlabel[3],
                                         str(bbox_extend[2]), str(bbox_extend[1]), str(bbox_extend[0]),
                                         str(other_location[0]), str(other_location[1]), str(other_location[2]),
                                         str(bbox_delta), str(other_vehilcle_label[1])])
                                # else:
                                #     # tmp_labels.append(['DontCare','0','0',str(alpha),tmp_2Dlabel[0],tmp_2Dlabel[1],tmp_2Dlabel[2],tmp_2Dlabel[3],
                                #     tmp_labels.append(['Car','0','0',str(alpha),0,0,0,0,
                                #                         str(bbox_extend[2]), str(bbox_extend[1]), str(bbox_extend[0]),
                                #                         str(other_location[0]), str(other_location[1]), str(other_location[2]), str(bbox_delta), str(other_vehilcle_label[1])])

                    if len(tmp_labels) == 0:
                        tmp_labels.append(
                            ['DontCare', '-1', '-1', '-10', '522.25', '202.35', '547.77', '219.71', '-1', '-1', '-1',
                             '-1000', '-1000', '-1000', '-10', '-10'])
                    #raw_data_path: record, _frame, ego_vehicle在label中的信息， tmp_labels:所有other_vehicle的extend,location和sensor夹角信息，
                    #pointCloud:将点云源文件中的数据转换为o3d可以处理的类型, index:0, camera_id, calib_infor_list:校准矩阵
                    img_path = write_labels(raw_data_path, _frame, ego_vehicle_label, tmp_labels, pointCloud, index,
                                            camera_id, calib_info_list)
                    # print(label2d)
                    show_flag = False
                    if show_flag:
                        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                        o3d.visualization.draw_geometries(bboxes + [pointCloud, mesh] + tmp_p, width=960, height=640,
                                                          point_show_normal=True)

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        img = np.array(o3d.io.read_image(img_path))
                        plt.imshow(img)
                        for rect in label2d:
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            rect = plt.Rectangle((rect[0], rect[1]), width, height, fill=False, color='b')
                            ax.add_patch(rect)
                        plt.show()

                    # plt.close()
                    # o3d.visualization.draw_geometries([img],width=1242,height=375)
                    # exit()
            elif 'traffic.traffic_light' in ego_vehicle_label[0]:
                try:
                    lidar_center, lidar_rotation,lidar_rotation_test = get_sensor_transform(ego_vehicle_label[1],labels,sensor='lidar')
                except:
                    continue

                calib_info_list = []
                for index in range(4):
                    calib_info_list.append(' '.join(post(camera_intrinsic_matrix, index)))
                tmp_info = ['R0_rect:'] + list(map(str, [1, 0, 0, 0, 1, 0, 0, 0, 1]))
                calib_info_list.append(' '.join((copy.deepcopy(tmp_info))))
                tmp_info[0] = 'Tr_imu_to_velo:'
                calib_info_list.append(' '.join(tmp_info))
                calib_info_list.append(' '.join(tmp_info))

                camera_info_list = get_sensor_transform(ego_vehicle_label[1], labels, sensor='camera')
                lidar_raw_data_file = get_ego_lidar_data_path(ego_vehicle_label, _frame[:-3])
                pointCloud = get_raw_lidar_data(lidar_raw_data_file, lidar_rotation_test)
                for index, camera in enumerate(camera_info_list):
                    camera_center, camera_rotation, camera_id = camera
                    velo_to_cam_matrix = get_matrix_from_origin_to_target(lidar_center,lidar_rotation,camera_center,camera_rotation)
                    post_velo_to_cam = process_matrix(velo_to_cam_matrix)
                    calib_info_list[-2] = ' '.join(post_velo_to_cam)

                    calib = get_calib_file(raw_data_path, _frame, ego_vehicle_label, index, calib_info_list)
                    tmp_pointCloud = process_pcd(pointCloud,calib)

                    ego_vehicle_location, ego_vehicle_rotation = get_traffic_light_transform(ego_vehicle_label)
                    tmp_labels,bboxes = [],[]
                    label2d = []
                    label2point = [[],[]]
                    tmp_p = []
                    camera_2Dlabel_file = get_ego_camera_2Dlabel_path(ego_vehicle_label, camera_id, _frame[:-3])
                    try:
                        camera_2Dlabel = np.loadtxt(camera_2Dlabel_file).reshape(-1,5)
                    except:
                        camera_2Dlabel = np.array([])
                    for other_vehilcle_label in labels:
                        if 'vehicle' in other_vehilcle_label[0]:
                            if ego_vehicle_label[1] == other_vehilcle_label[1]:# or ego_vehicle_label[1] != '247' or other_vehilcle_label[1] not in ['229','216']:
                                continue
                            # object_detected_number = judge_in_ROI_numbers(other_vehilcle_label,AD_vehicles_location,_frame[:-3])
                            object_detected_number = 0
                            # exit()
                            other_location, other_rotation = get_vehicle_transform(other_vehilcle_label)
                            other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,lidar_center,lidar_rotation)
                            tmp = tmp_pointCloud.crop(other_bbox)
                            other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,lidar_center,lidar_rotation,True)
                            tmp2 = tmp_pointCloud.crop(other_bbox)

                            #存放vehicle/label文件夹下每个AV可以看到的other vechicle的min_x,min_y,max_x,max_y
                            tmp_2Dlabel = []
                            for _label in camera_2Dlabel:
                                if other_vehilcle_label[1] == str(int(_label[0])):
                                    tmp_2Dlabel = _label[1:]
                                    if tmp_2Dlabel[0] < 0:
                                        tmp_2Dlabel[0] = 0
                                    if tmp_2Dlabel[1] < 0:
                                        tmp_2Dlabel[1] = 0
                                    if tmp_2Dlabel[2] > 1242:
                                        tmp_2Dlabel[2] = 1242
                                    if tmp_2Dlabel[3] > 375:
                                        tmp_2Dlabel[3] = 375
                                    break
                            if len(tmp.points) >= 10 and len(tmp_2Dlabel) != 0:
                            # if True:
                                # camera_label = transform_lidar_to_camera(other_bbox.get_box_points(),camera_intrinsic_matrix, velo_to_cam_matrix)
                                pcd = o3d.geometry.PointCloud()
                                tmp_pcd = np.array(np.array(other_bbox.get_box_points()), dtype=np.dtype('f4'))
                                pcd.points = o3d.utility.Vector3dVector(tmp_pcd)
                                pcd.paint_uniform_color([1,0,0])
                                tmp_p.append(pcd)

                                label2d.append(tmp_2Dlabel)
                                # camera_label = [np.array(i)[0][0] for i in camera_label]
                                other_location = other_bbox.get_center()
                                bboxes.append(other_bbox)
                                # print(other_location,float(other_vehilcle_label[-1]))
                                other_location -= np.array([0, 0, float(other_vehilcle_label[-1])])
                                # other_location = transform_vehicle_to_lidar(other_location, matrix_to_ego)
                                alpha = - bbox_delta - math.atan(other_location[0] / other_location[2]) - np.pi


                                def process_theta(alpha):
                                    while alpha > np.pi:
                                        alpha -= np.pi * 2
                                    while alpha < -np.pi:
                                        alpha += np.pi * 2
                                    return alpha


                                alpha = process_theta(alpha)
                                # print(-bbox_delta-np.pi+lidar_rotation[2]+np.pi/2,np.radians(float(other_vehilcle_label[7])))
                                bbox_delta = process_theta(-bbox_delta - np.pi)
                                if len(tmp2.points) > 10:
                                    size = 0
                                    if len(tmp2.points) + other_location[0] < 250:
                                        size = 1
                                    if len(tmp2.points) + other_location[0] < 125:
                                        size = 2
                                    tmp_labels.append(
                                        ['Car', str(object_detected_number), str(size), str(alpha), tmp_2Dlabel[0],
                                         tmp_2Dlabel[1], tmp_2Dlabel[2], tmp_2Dlabel[3],
                                         str(bbox_extend[2]), str(bbox_extend[1]), str(bbox_extend[0]),
                                         str(other_location[0]), str(other_location[1]), str(other_location[2]),
                                         str(bbox_delta), str(other_vehilcle_label[1])])

                    if len(tmp_labels) == 0:
                        tmp_labels.append(
                            ['DontCare', '-1', '-1', '-10', '522.25', '202.35', '547.77', '219.71', '-1', '-1', '-1',
                             '-1000', '-1000', '-1000', '-10', '-10'])
                    img_path = write_labels(raw_data_path, _frame, ego_vehicle_label, tmp_labels, pointCloud, index,
                                            camera_id, calib_info_list)
                    # print(label2d)
                    show_flag = False
                    if show_flag:
                        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                        o3d.visualization.draw_geometries(bboxes + [pointCloud, mesh] + tmp_p, width=960, height=640,
                                                          point_show_normal=True)

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        img = np.array(o3d.io.read_image(img_path))
                        plt.imshow(img)
                        for rect in label2d:
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            rect = plt.Rectangle((rect[0], rect[1]), width, height, fill=False, color='b')
                            ax.add_patch(rect)
                        plt.show()
            else:
                continue

    img_list_file_path = (COOK_DATA_PATH / raw_data_path.stem / 'img_list.txt').as_posix()
    frame_hz_alt = RAW_DATA_FREQ_ALT  # for other img_list with higher frame_hz
    frames = [frame[:-4] for frame in frames[frame_start:frame_end:frame_hz_alt]]
    np.savetxt(img_list_file_path, np.array(frames), fmt='%s', delimiter=' ')
    # training: 0:510:3
    # testing: 510:1100:1
    # if not os.path.exists(global_label_file_path):
    #     os.makedirs(global_label_file_path[:-13])
    # np.savetxt(global_label_file_path, np.array(frames),fmt='%s', delimiter=' ')
