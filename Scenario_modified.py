#!/usr/bin/env python3
import os
import sys
import time
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
import random
import weakref
import logging
import argparse
# import mayavi.mlab

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from carla import ColorConverter as cc

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
ApplyVehicleControl = carla.command.ApplyVehicleControl
Attachment = carla.AttachmentType

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    sys.path.append(Path(CARLA_PATH, 'PythonAPI/carla').expanduser().as_posix())
    sys.path.append(Path(CARLA_PATH, 'PythonAPI/examples').expanduser())
except IndexError:
    pass

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

from utils.get2Dlabel import ClientSideBoundingBoxes


class Args(object):
    def __init__(self, argv=None):
        # client information
        self.host = '127.0.0.1'
        self.port = 2000
        # traffic manager默认端口：8000
        self.tm_port = 8000
        self.time_out = 10.0
        # 设置帧数，每秒20帧，每一帧的时间为0.05
        self.fixed_delta_seconds = 0.05

        # map information
        self.map_name = TOWN_MAP
        # 设置观察者视角坐标, 包括localtion, rotation(pitch,yaw,rotate)
        self.spectator_point = [(100, 150, 150), (-60, 90, 0)]
        # ROI正方形的四个点坐标
        self.ROI = [[-140, -150], [-140, 140], [150, 140], [150, -150]]
        #traffic_light区域
        self.traffic_ROI = [-140, 150, -150, 140]
        # 如果pretrain_model为false,这些为spawn_point
        # 四个坐标分别为x的范围，y的范围，一共有6个ROI
        self.initial_spawn_ROI = [[-80, -70, 0, 100],
                                  [-90, -80, -120, 0],
                                  [-150, -90, 0, 10],
                                  [0, 10, 0, 100],
                                  [-10, 0, -150, 0],
                                  [0, 150, -10, 0]]
        self.additional_spawn_ROI = [[-80, -70, 90, 100],
                                     [-90, -80, -120, -100],
                                     [-150, -140, 0, 10],
                                     [0, 10, 80, 135],
                                     [-10, 0, -150, -120],
                                     [140, 150, -10, 0]]

        # server information
        # self.fixed_delta_seconds = 0.1
        # agent information
        self.agent = 'BehaviorAgent'
        # record information
        self.sync = True
        # 当地时间四位年，月日，24小时制时分
        self.time = time.strftime("%Y_%m%d_%H%M", time.localtime())
        # /用来连接路径
        self.recorder_filename = (LOG_PATH / ('record' + self.time + '.log')).as_posix()
        # self.recorder_filename = os.getcwd() + '/' + 'log/record' + self.time + '.log'
        if argv:
            self.task = argv[1]
            if self.task == 'record' and len(argv) == 4:
                self.hd_id = [int(id) for id in argv[2].split(',')]  # 'hd' for 'human driver'
                self.av_id = [int(id) for id in argv[3].split(',')]  # 'av' for 'autonomous vehicle'
            if len(argv) > 2 and self.task == 'replay':
                self.recorder_filename = self.recorder_filename[:-13] + sys.argv[2] + '.log'
            elif len(argv) == 2 and self.task == 'replay':
                record_path = os.listdir(self.recorder_filename[:-24])
                self.recorder_filename = self.recorder_filename[:-24] + record_path[-1]

        self.time_factor = 1.0
        self.camera = 0
        self.start = 0
        self.duration = 0
        # raw data information
        self.raw_data_path = RAW_DATA_PATH / ('record' + self.time)
        # self.raw_data_path = 'tmp/record' + self.time + '/'
        self.image_width = 1242
        self.image_height = 375
        # 设置点云采集的水平视角范围
        self.VIEW_FOV = 90

        # np.identity(3)创建3*3方阵
        self.calibration = np.identity(3)
        self.calibration[0, 2] = self.image_width / 2.0
        self.calibration[1, 2] = self.image_height / 2.0
        self.calibration[0, 0] = self.calibration[1, 1] = self.image_width / (
                    2.0 * np.tan(self.VIEW_FOV * np.pi / 360.0))

        self.sample_frequence = 1  # 20frames/s


class Map(object):
    def __init__(self, args):
        self.pretrain_model = True
        self.client = carla.Client(args.host, args.port)
        self.world = self.client.get_world()
        self.initial_spectator(args.spectator_point)
        self.tmp_spawn_points = self.world.get_map().get_spawn_points()
        self.traffic_ROI = args.traffic_ROI
        if self.pretrain_model:
            self.initial_spawn_points = self.tmp_spawn_points
        else:
            self.initial_spawn_points = self.check_spawn_points(args.initial_spawn_ROI)
        self.additional_spawn_points = self.check_spawn_points(args.additional_spawn_ROI)
        self.destination = self.init_destination(self.tmp_spawn_points, args.ROI)
        self.ROI = args.ROI
        try:
            self.hd_id = args.hd_id
            self.av_id = args.av_id
        except:
            self.hd_id = []
            self.av_id = []

    def initial_spectator(self, spectator_point):
        spectator = self.world.get_spectator()
        # 设置观察者时间的transform，包括location和rotation. 前面定义了(x,y,z),(pitch,yaw,roll)
        spectator_point_transform = carla.Transform(carla.Location(spectator_point[0][0],
                                                                   spectator_point[0][1],
                                                                   spectator_point[0][2]),
                                                    carla.Rotation(spectator_point[1][0],
                                                                   spectator_point[1][1],
                                                                   spectator_point[1][2]))
        spectator.set_transform(spectator_point_transform)

    # 检查啊map.get_spawn_pionts中的点是不是在ROI，是的话返回，不是的话抛弃
    def check_spawn_points(self, check_spawn_ROI):
        tmp_spawn_points = []
        tmpx, tmpy = [], []
        for tmp_transform in self.tmp_spawn_points:
            tmp_location = tmp_transform.location
            for edge in check_spawn_ROI:
                if tmp_location.x > edge[0] and tmp_location.x < edge[1] and tmp_location.y > edge[
                    2] and tmp_location.y < edge[3]:
                    tmp_spawn_points.append(tmp_transform)
                    tmpx.append(tmp_location.x)
                    tmpy.append(tmp_location.y)
                    continue
        # self.plot_points(tmpx,tmpy)
        return tmp_spawn_points

    def check_traffic_light_points(trafficlights, ROI):
        checked_tl = []
        for traffic_light in trafficlights:
            tl_location = traffic_light.location
            if tl_location.x > ROI[0] and tl_location.x < ROI[1] and tl_location.y > ROI[2] and tl_location.y < ROI[3]:
                checked_tl.append(traffic_light)
        return checked_tl

    def plot_points(self, tmpx, tmpy, tfx, tfy):
        plt.figure(figsize=(11, 11))
        # 子表格为1*1，选择第一个。 坐标轴范围x:[-50,250];y:[50,350]
        ax = plt.subplot(111)
        ax.axis([-50, 250, 50, 350])
        ax.scatter(tmpx, tmpy,c='red')
        ax.scatter(tfx,tfy,c='green')
        for index in range(len(tmpx)):
            # 添加一个text备注，坐标为(tmpx,tmpy)，内容为index
            ax.text(tmpx[index], tmpy[index], index)
        for index in range(len(tfx)):
            ax.text(tfx[index], tfy[index], index)
        plt.show()

    def init_destination(self, spawn_points, ROI):
        destination = []
        tmpx, tmpy = [], []
        for p in spawn_points:
            if not self.inROI([p.location.x, p.location.y], ROI):
                destination.append(p)
                tmpx.append(p.location.x)
                tmpy.append(p.location.y)
        # self.plot_points(tmpx,tmpy)
        return destination

    # 向量ac和向量bc的插技
    # 通过这两个向量的差积判断这个点是不是在四边形内
    # https://zhuanlan.zhihu.com/p/94758998
    def sign(self, a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    # 判断x是否落在ROI指定的那六个区域中，
    def inROI(self, x, ROI):
        d1 = self.sign(x, ROI[0], ROI[1])
        d2 = self.sign(x, ROI[1], ROI[2])
        d3 = self.sign(x, ROI[2], ROI[3])
        d4 = self.sign(x, ROI[3], ROI[0])

        # 相同符号表示在正方形内
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
        return not (has_neg and has_pos)

    def shuffle_spawn_points(self, spawn_points, start=False):
        # random.shuffle(spawn_points)
        if self.pretrain_model:
            # self.av_id = [4,5,27,20,97,22,14,77,47]
            # self.hd_id = [19,21,29,31,44,48,87,96] + [i for i in range(50,70)]
            # connected autonomous vehicle, human drive
            cav = [spawn_points[i] for i in self.av_id]
            hd = [spawn_points[i] for i in self.hd_id]
            if len(cav) == 0 and len(hd) == 0:
                return spawn_points[:60], spawn_points[-20:]
            else:
                return hd, cav


class Server(object):
    def __init__(self):
        pass


class Vehicle_Agent(BehaviorAgent):
    def __init__(self, vehicle):
        BehaviorAgent.__init__(self, vehicle)

    def planning_ang_control(self):
        pass


# agent线程：一个vehicle是一个包含vehicle_actor的BehaviorAgent,设置这个agent的全局，局部路径控制
class CAVcontrol_Thread(Thread):
    #   继承父类threading.Thread
    def __init__(self, vehicle, world, destination, num_min_waypoints, control):
        Thread.__init__(self)
        self.v = vehicle
        self.w = world
        self.d = destination
        self.n = num_min_waypoints
        self.c = control
        self.start()

    def run(self):
        #   把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.control = None
        self.v.update_information(self.w)
        if len(self.v.get_local_planner().waypoints_queue) < self.n:
            self.v.reroute(self.d)
        speed_limit = self.v.vehicle.get_speed_limit()
        self.v.get_local_planner().set_speed(speed_limit)
        self.control = self.c(self.v.vehicle.id, self.v.run_step())

    def return_control(self):
        #   threading.Thread.join(self) # 等待线程执行完毕
        self.join()
        try:
            return self.control
        except Exception:
            print('This is an issue')


# init过程：创建camera,lidarsensor，设置相应sensor的属性，同时append到sensor_attribute中。
# 同时，它是一个进程，上面运行start(), 这里运行run函数，设置lidar,camera存储raw_data的路径，用call_back函数_parse_image存储产生的raw_data
class CAVcollect_Thread(Thread):
    def __init__(self, parent_id, sensor_attribute, sensor_transform, args):
        Thread.__init__(self)
        self.recording = False
        self.args = args
        # 相机畸变参数
        gamma_correction = 2.2
        # Attachment = carla.AttachmentType
        self.client = carla.Client(self.args.host, self.args.port)
        world = self.client.get_world()
        self.sensor = None
        self._parent = world.get_actor(parent_id)
        self._camera_transforms = sensor_transform  # (sensor_transform, Attachment.Rigid)
        bp_library = world.get_blueprint_library()
        bp = bp_library.find(sensor_attribute[0])
        if sensor_attribute[0].startswith('sensor.camera'):
            bp.set_attribute('image_size_x', str(self.args.image_width))
            bp.set_attribute('image_size_y', str(self.args.image_height))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            for attr_name, attr_value in sensor_attribute[3].items():
                bp.set_attribute(attr_name, attr_value)
        elif sensor_attribute[0].startswith('sensor.lidar'):
            bp.set_attribute('range', '100')
            bp.set_attribute('channels', '64')
            bp.set_attribute('points_per_second', '2240000')
            # 注意这个频率和fixed_delta_seconds一样
            bp.set_attribute('rotation_frequency', '20')
            bp.set_attribute('sensor_tick', str(0.05))
            bp.set_attribute('dropoff_general_rate', '0.0')
            bp.set_attribute('dropoff_intensity_limit', '1.0')
            bp.set_attribute('dropoff_zero_intensity', '0.0')
            # bp.set_attribute('noise_stddev', '0.0')
        sensor_attribute.append(bp)
        self.sensor_attribute = sensor_attribute

    def run(self):
        self.set_sensor()

    def set_sensor(self):
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensor_attribute[-1],
            self._camera_transforms[0],
            attach_to=self._parent)
        # attachment_type=self._c#amera_transforms[1])
        # vehicle.tesla.model3_448; sensor.camera.rgb_450 sensor.lidar.ray_cast_451存放的rgb和ply的raw_data
        filename = Path(self.args.raw_data_path,
                        '%s_%d' % (self._parent.type_id, self._parent.id),
                        '%s_%d' % (self.sensor.type_id, self.sensor.id)
                        ).as_posix()
        # filename = self.args.raw_data_path + \
        #             self._parent.type_id + '_' + str(self._parent.id) + '/' + \
        #             self.sensor.type_id + '_' + str(self.sensor.id)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CAVcollect_Thread._parse_image(weak_self, image, filename))
        # self.sensor.stop()
        # print(filename)

    def get_sensor_id(self):
        self.join()
        return self.sensor.id
    #listen对应的callback函数
    @staticmethod
    def _parse_image(weak_self, image, filename):
        self = weak_self()
        if image.frame % self.args.sample_frequence != 0:
            return
        if self.sensor.type_id.startswith('sensor.camera'):
            image.convert(self.sensor_attribute[1])
            image.save_to_disk(filename + '/%010d' % image.frame)  # 0000002057.png
        else:
            image.save_to_disk(filename + '/%010d' % image.frame)  # 0000002059.ply


class Scenario(object):
    def __init__(self, args):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.time_out)
        self.world = self.client.load_world(args.map_name)
        self.traffic_manager = self.client.get_trafficmanager(args.tm_port)
        self.map = Map(args)
        self.recording_rawdata = False
        # agent information
        self.HD_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        self.CAV_blueprints = self.world.get_blueprint_library().filter('vehicle.tesla.*')
        # sensor information: cc:carla.ColorConverter-->
        self.sensor_attribute = [['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                                 # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                                 # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                                 # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                                 # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                                 # ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,'Camera Semantic Segmentation (CityScapes Palette)', {}],
                                 # ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,'Camera Semantic Segmentation (CityScapes Palette)', {}],
                                 # ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
                                 ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]]
        self.sensor_transform = [(carla.Transform(carla.Location(x=0, z=2.5)), Attachment.Rigid),
                                 # (carla.Transform(carla.Location(x=1.6, z=2.5),carla.Rotation(pitch=-10,yaw=90)), Attachment.Rigid),
                                 # (carla.Transform(carla.Location(x=1.6, z=2.5),carla.Rotation(pitch=-10,yaw=180)), Attachment.Rigid),
                                 # (carla.Transform(carla.Location(x=1.6, z=2.5),carla.Rotation(pitch=-10,yaw=270)), Attachment.Rigid),
                                 # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                                 # (carla.Transform(carla.Location(x=1.6, z=2.5)), Attachment.Rigid),
                                 # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                                 (carla.Transform(carla.Location(x=0, z=2.5)), Attachment.Rigid)]
        self.args = args
        weak_self = weakref.ref(self)
        #world.on_tick()返回当前帧号，在里面的是callback函数，使用每一帧的world_snapshot和一个self的weakref作为输入
        self.world.on_tick(lambda world_snapshot: self.on_world_tick(weak_self, world_snapshot))

    @staticmethod
    def parse_transform(transform):
        return [transform.location.x, transform.location.y, transform.location.z, transform.rotation.roll,
                transform.rotation.pitch, transform.rotation.yaw]

    @staticmethod
    def parse_bounding_box(bounding_box):
        return [bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z, bounding_box.location.z]

    # 为world产生的每一帧创建label文件夹，里面记录了vehicle和sensor的数据
    # 数据包括：type_id,id,x,y,z,roll,yaw,pitch; vehicle还有boudingbox的四个值，sensor还有[0,0,0],parent_id
    @staticmethod
    def on_world_tick(weak_self, world_snapshot):
        self = weak_self()
        if world_snapshot.frame % self.args.sample_frequence != 0:
            return
        if self.args.task == 'replay':  # or world_snapshot.frame % 2 == 0:
            return
        actors = self.world.get_actors()
        traffic_light, vehicles, sensors, CAV_vehicles = [], [], [], []
        for actor in actors:
            #type_id, id, x,y,z, roll,pitch,yaw + [0,0,0] traffic_id/extend,x,y,z,location.z/[0,0,0]
            str_actor = [str(actor.type_id), actor.id] + Scenario.parse_transform(actor.get_transform())
            # if 'lidar' in actor.type_id or 'rgb' in actor.type_id:
            #     print(str(actor.type_id),actor.get_transform().rotation.pitch,actor.get_transform().rotation.roll)
            if 'traffic_light' in actor.type_id:
                str_actor += [0,0,0]+[actor.id]
                traffic_light.append(str_actor)
            if 'vehicle' in actor.type_id:
                str_actor += Scenario.parse_bounding_box(actor.bounding_box)
                vehicles.append(str_actor)
            elif 'sensor' in actor.type_id:
                str_actor += [0, 0, 0] + [actor.parent.id]
                sensors.append(str_actor)
        # actors是一个有vechicles和sensors组成的np.array,被存在label文件中
        actors = np.array(traffic_light+vehicles + sensors)
        _label_path = Path(self.args.raw_data_path, 'label')
        _label_path.mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(self.args.raw_data_path+'label'):
        #     os.makedirs(self.args.raw_data_path+'label')
        if len(actors) != 0:
            # label/0000002055.txt
            _filename = (_label_path / ('%010d.txt' % (world_snapshot.frame))).as_posix()
            np.savetxt(_filename, actors, fmt='%s', delimiter=' ')
            # np.savetxt(self.args.raw_data_path + '/label/%010d.txt' % world_snapshot.frame, actors, fmt='%s', delimiter=' ')

        ##sensor relation是一个字典，key为sensor的parent_id, sensor_id
        # 2D bounding box
        # actor_list: vehicles
        #这部分实在raw_data/具体parent文件夹下产生另一个label文件夹，里面放的parent sensor感受到的周围车辆的bounding box的全局坐标
        vehicles = self.world.get_actors().filter('vehicle.*')
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        traffic_light = traffic_lights[4]
        print('traffic light id:',traffic_light.id)
        #找到每个带sensor的vehicle,然后将sensor感受到的范围内的所有车辆的boundingbox信息存到车辆对应的文件夹中

        for vehicle in vehicles:
            if str(vehicle.id) in self.sensor_relation.keys():
                if hasattr(self, "sensor_relation"):
                    sensor_list = self.sensor_relation[str(vehicle.id)]
                    print('line464',sensor_list)
            else:
                continue
            calib_info = []
            for sensor_id in sensor_list:
                sensor = self.world.get_actor(sensor_id)
                if 'lidar' in sensor.type_id:
                    lidar = self.world.get_actor(sensor_id)
            # 存储摄像头范围内的所有车两的bounding box raw_data/record_XXXX_XXXX/vechicle.tesla.model3_448/sensor.camera.rgb_450_label
            # 记录的是sensor_id,以及这个sensor看到的车辆的bounding box, 中间经过了从camera坐标系到世界坐标系的转换
            for sensor_id in sensor_list:
                sensor = self.world.get_actor(sensor_id)
                if 'rgb' in sensor.type_id:
                    sensor.calibration = self.args.calibration
                    tmp_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, sensor)
                    image_label_path = Path(self.args.raw_data_path,
                                            vehicle.type_id + '_' + str(vehicle.id),
                                            sensor.type_id + '_' + str(sensor.id)
                                            ).as_posix()
                    #/record2021_0624_2149/vehicle.tesla.model3_448/sensor.camera.rgb_450_label
                    # image_label_path = self.args.raw_data_path + \
                    #                     vehicle.type_id + '_' + str(vehicle.id) + '/' + \
                    #                     sensor.type_id + '_' + str(sensor.id)

                    if not os.path.exists(image_label_path + '_label'):
                        os.makedirs(image_label_path + '_label')
                    if len(tmp_bboxes) != 0:
                        np.savetxt(image_label_path + '_label/%010d.txt' % world_snapshot.frame, tmp_bboxes, fmt='%s',
                                   delimiter=' ')
                # lidar_to_camera_matrix = ClientSideBoundingBoxes.get_lidar_to_camera_matrix(lidar, sensor)
                # calib_info.append(lidar_to_camera_matrix)
        # print(hasattr(self,'sensor_relation'))
        if hasattr(self, "sensor_relation"):
            # print(self.sensor_relation)
            # print(traffic_light.id)
            # print(str(traffic_light.id) in self.sensor_relation.keys()) #false

            # if str(91) in self.sensor_relation.keys():
            if str(traffic_light.id) in self.sensor_relation.keys(): #here is false
                print('line 449',traffic_light.id)
                sensor_list = self.sensor_relation[str(traffic_light.id)]
                print('line501',sensor_list)
                for sensor_id in sensor_list:
                    sensor = self.world.get_actor(sensor_id)
                    print(sensor.type_id)
                    if 'rgb' in sensor.type_id:
                        sensor.calibration = self.args.calibration
                        #记录的是sensor_id,以及这个sensor看到的车辆的bounding box, 中间经过了从camera坐标系到世界坐标系的转换 (extend.x...)-->(x,y,z)
                        tmp_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, sensor)
                        image_label_path = Path(self.args.raw_data_path,
                                                traffic_light.type_id + '_' + str(traffic_light.id),
                                                sensor.type_id+'_'+str(sensor.id)).as_posix()
                        print(image_label_path)
                        if not os.path.exists(image_label_path + '_label'):
                            os.makedirs(image_label_path + '_label')
                        if len(tmp_bboxes) != 0:
                            np.savetxt(image_label_path + '_label/%010d.txt' % world_snapshot.frame, tmp_bboxes,
                                       fmt='%s',
                                       delimiter=' ')

    def look_for_spawn_points(self, args):
        try:
            # 设置同步模式，同时plt画出initial_spawn_point
            self.start_look(args)
            if not args.sync or not self.synchronous_master:
                self.world.wait_for_tick()
            else:
                start = self.world.tick()
            while True:
                if args.sync and self.synchronous_master:
                    # world.tick()返回frame的ID
                    now = self.run_step()
                    if (now - start) % 1000 == 0:
                        print('Frame ID:' + str(now))
                else:
                    self.world.wait_for_tick()
        finally:
            try:
                print('stop from frameID: %s.' % now)
            finally:
                pass
            self.stop_look(args)
            pass

    # 设置同步模式，用plt画出spawn_point的位置
    def start_look(self, args):
        self.synchronous_master = False
        if args.sync:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                self.synchronous_master = True
                self.traffic_manager.set_synchronous_mode(True)
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = args.fixed_delta_seconds
                self.world.apply_settings(settings)
            else:
                self.synchronous_master = False
                print('synchronous_master is False.')
        tmpx, tmpy = [], []
        tfx, tfy = [], []
        for tmp_transform in self.map.initial_spawn_points:
            tmp_location = tmp_transform.location
            tmpx.append(((tmp_location.x - 100) * -1) + 100)
            tmpy.append(tmp_location.y)
        all_traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        all_traffic_points = []
        for traffic_light in all_traffic_lights:
            all_traffic_points.append(traffic_light.get_transform())
        # tmp_traffic_points = self.map.check_traffic_light_points(all_traffic_points, map.traffic_ROI )
        for tmp_transform in all_traffic_points:
            tmp_location = tmp_transform.location
            tfx.append(((tmp_location.x)))
            tfy.append(tmp_location.y)
        # 在这里画出spawnpoint在地图中的位置
        self.map.plot_points(tmpx, tmpy,tfx,tfy)

    def stop_look(self, args):
        # 设置server与client 为原始模式，要设置settings，fix_delta_seconds，synchronous_mode,trafficmanager
        print(args.sync)
        if args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        self.world = self.client.reload_world()
        self.map.initial_spectator(args.spectator_point)

    def generate_data(self, args):
        self.recording_rawdata = True
        try:
            self.start_record(args)
            if not args.sync or not self.synchronous_master:
                self.world.wait_for_tick()
            else:
                start = self.world.tick()
                if self.dynamic_weather:
                    self.weather.tick(1)
                    self.world.set_weather(self.weather.weather)
                print('start from frameID: %s.' % start)
            while True:

                if args.sync and self.synchronous_master:
                    time.sleep(1)
                    now = self.run_step()
                    if (now - start) % 1000 == 0:
                        print('Frame ID:' + str(now))
                    #     self.add_anget_and_vehicles()
                else:
                    self.world.wait_for_tick()
        finally:
            try:
                print('stop from frameID: %s.' % now)
            finally:
                pass
            self.stop_record(args)
            pass

    # 设置天气，同步模式，生成.log文件
    def start_record(self, args):
        self.synchronous_master = False
        self.dynamic_weather = False
        if self.dynamic_weather:
            from dynamic_weather import Weather
            w = self.world.get_weather()
            w.precipitation = 80
            weather = Weather(w)
            self.weather = weather
        if args.sync:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                self.synchronous_master = True
                self.traffic_manager.set_synchronous_mode(True)
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = args.fixed_delta_seconds
                self.world.apply_settings(settings)
            else:
                self.synchronous_master = False
                print('synchronous_master is False.')
        # if not os.path.exists('log'):
        #     os.mkdir('log')
        #     print('mkdir log finished.')
        # recorder: https://carla.readthedocs.io/en/latest/adv_recorder/
        print("Recording on file: %s" % self.client.start_recorder(args.recorder_filename))
        self.agent_list = []
        self.sensor_relation = {}
        self.sensor_thread = []
        # 产生HD，CAV的spawnpoint
        HD_spawn_points, CAV_spawn_points = self.map.shuffle_spawn_points(self.map.initial_spawn_points, start=True)
        # print(len(CAV_spawn_points))
        # 分两部分，第一部分用来产生vehicle actor,又分为产生HD和CAV两部分,第二部分用来产生sensor actor (CAV全是tesla)
        self.HD_agents = self.spawn_actorlist('vehicle', self.HD_blueprints, HD_spawn_points)
        print(len(self.HD_agents))
        #CAV中的id_list包括AV的id和sensor的id
        self.CAV_agents = self.spawn_actorlist('vehicle', self.CAV_blueprints, CAV_spawn_points)

        self.traffic_light_agent = self.spawn_actorlist('traffic_light')

    def stop_record(self, args):
        if args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.CAV_agents + self.HD_agents])
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.camera_list])
        print('\ndestroying %d vehicles' % len(self.CAV_agents + self.HD_agents))
        self.sensor_list = []
        for sensor in self.sensor_relation.values():
            self.sensor_list += sensor
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
        print('\ndestroying %d sensors' % len(self.sensor_list))
        self.client.stop_recorder()
        print("Stop recording")

    # 分两部分，第一部分用来产生vehicle actor,又分为产生HD(没有tesla),第二部分用来产生CAV和sensor actor，返回所有车，senor的id_list
    def spawn_actorlist(self, actor_type, agent_blueprint=None, spawn_points=None, parent_agent=None):
        bacth_spawn = []
        id_list = []
        if actor_type == 'traffic_light':
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
            # traffic_light = random.choice(traffic_lights)
            traffic_light = traffic_lights[4]
            id_list.append(traffic_light.id)
            #第一个是rgb camera, 第二个是lidar
            sensor_transform = [(carla.Transform(carla.Location(x=0, z=5),carla.Rotation(pitch=-10,yaw=120)), Attachment.Rigid),
                                     # (carla.Transform(carla.Location(x=1.6, z=2.5),carla.Rotation(pitch=-10,yaw=180)), Attachment.Rigid),
                                     # (carla.Transform(carla.Location(x=1.6, z=2.5),carla.Rotation(pitch=-10,yaw=270)), Attachment.Rigid),
                                    (carla.Transform(carla.Location(x=0, z=5),carla.Rotation(pitch=-10, yaw=120)), Attachment.Rigid)]
            tmp_sensor_id_list = self.spawn_actorlist('sensor', self.sensor_attribute,
                                                      sensor_transform, traffic_light.id)
            print(tmp_sensor_id_list)
            self.sensor_relation[str(traffic_light.id)] = tmp_sensor_id_list
            print('line696',str(traffic_light.id))
            print('line697', self.sensor_relation)
            print('line698', self.sensor_relation.keys())


            # for traffic_light in traffic_lights:
            #     id_list.append(traffic_light.id)
            #     tmp_sensor_id_list = self.spawn_actorlist('sensor', self.sensor_attribute,
            #                                               self.sensor_transform, traffic_light.id)
            #     self.sensor_relation[str(traffic_light.id)] = tmp_sensor_id_list

        elif actor_type == 'vehicle':

            if not random.choice(agent_blueprint).id.startswith('vehicle.tesla'):
                # HD_agents
                # print(len(spawn_points))
                for n, transform in enumerate(spawn_points):
                    blueprint = random.choice(agent_blueprint)
                    while 'tesla' in blueprint.id or 'crossbike' in blueprint.id or 'low_rider' in blueprint.id:
                        blueprint = random.choice(agent_blueprint)
                    bacth_spawn.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
                for response in self.client.apply_batch_sync(bacth_spawn, False):
                    if response.error:
                        print(response.error)
                        logging.error(response.error)
                    else:
                        tmp_vehicle = self.world.get_actor(response.actor_id)
                        if tmp_vehicle.bounding_box.extent.y < 0.6:
                            print(tmp_vehicle.bounding_box.extent.y)
                            tmp_vehicle.destroy()
                        # print(tmp_vehicle.bounding_box.extent.y)
                        if int(tmp_vehicle.attributes['number_of_wheels']) == 2:
                            tmp_vehicle.destroy()
                        else:
                            id_list.append(response.actor_id)
            elif random.choice(agent_blueprint).id.startswith('vehicle.tesla'):
                # CAV_agents
                for n, transform in enumerate(spawn_points):
                    blueprint = random.choice(agent_blueprint)
                    bacth_spawn.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
                # 每个vehicle上以后两个sensor,一个是camera,一个是lidar所以有一个tmp_sensor_id_list
                # 每个sensor_relation是一个字典，key是vehicle_id, value是lidar和camera组成的list
                for response in self.client.apply_batch_sync(bacth_spawn, True):
                    if response.error:
                        logging.error(response.error)
                    else:
                        id_list.append(response.actor_id)
                        vehicle = self.client.get_world().get_actor(response.actor_id)
                        tmp_sensor_id_list = self.spawn_actorlist('sensor', self.sensor_attribute,
                                                                  self.sensor_transform, response.actor_id)
                        self.sensor_relation[str(response.actor_id)] = tmp_sensor_id_list
                        random.shuffle(self.map.destination)
                        tmp_agent = Vehicle_Agent(vehicle)
                        tmp_agent.set_destination(tmp_agent.vehicle.get_location(), self.map.destination[0].location,
                                                  clean=True)
                        self.agent_list.append(tmp_agent)
        elif actor_type == 'sensor':
            # sensor agents
            if spawn_points:
                sensor_trans = spawn_points
            else:
                sensor_trans = self.sensor_transform
            for index in range(len(self.sensor_attribute)):
                sensor_attribute = self.sensor_attribute[index]
                transform = sensor_trans[index]
                # 新的线程：创建camera,lidarsensor，设置相应sensor的属性，同时append到sensor_attribute中。
                # 同时，它是一个进程，运行run函数，设置lidar,camera存储raw_data的路径，用call_back函数_parse_image存储产生的raw_data
                #parent_agent传入的是parent_id
                tmp_sensor = CAVcollect_Thread(parent_agent, sensor_attribute, transform, self.args)
                tmp_sensor.start()
                self.sensor_thread.append(tmp_sensor)
                id_list.append(tmp_sensor.get_sensor_id())
        return id_list

    # 检查HD,CAV 车辆是不是在ROI中不是的话进行销毁
    def check_vehicle_state(self):
        for v_id in self.HD_agents:

            vehicle = self.world.get_actor(v_id)
            v_position = vehicle.get_transform().location
            if not (self.map.inROI([v_position.x, v_position.y], self.map.ROI)):
                vehicle.destroy()
                self.HD_agents.remove(v_id)

        for v_id in self.CAV_agents:
            vehicle = self.world.get_actor(v_id)
            v_position = vehicle.get_transform().location
            if not (self.map.inROI([v_position.x, v_position.y], self.map.ROI)):
                # for agent in self.agent_list:
                #     if agent.vehicle.id == v_id:
                #         self.agent_list.remove(agent)
                #         delete camera
                #         break
                for sensor_id in self.sensor_relation[str(v_id)]:
                    sensor = self.world.get_actor(sensor_id)
                    if sensor.is_listening:
                        print(sensor.id)
                        sensor.stop()
                    sensor.destroy()
                self.sensor_relation.pop(str(v_id))
                vehicle.destroy()
                self.CAV_agents.remove(v_id)

    def run_step(self):
        # batch_control = []
        # thread_list = []
        # num_min_waypoints = 21
        # for agent in self.agent_list:
        #     t = CAVcontrol_Thread(agent, self.world, self.map.destination, num_min_waypoints, ApplyVehicleControl)
        #     thread_list.append(t)
        # for t in thread_list:
        #     batch_control.append(t.return_control())
        # for response in self.client.apply_batch_sync(batch_control, False):
        #     if response.error:
        #         logging.error(response.error)
        if not self.map.pretrain_model:
            ##检查HD,CAV 车辆是不是在ROI中不是的话进行销毁
            self.check_vehicle_state()
        return self.world.tick()

    def add_anget_and_vehicles(self):
        HD_additional_spawn_points, CAV_additional_spawn_points = self.map.shuffle_spawn_points(
            self.map.additional_spawn_points)
        self.HD_agents += self.spawn_actorlist('vehicle', self.HD_blueprints, HD_additional_spawn_points)
        self.CAV_agents += self.spawn_actorlist('vehicle', self.CAV_blueprints, CAV_additional_spawn_points)

    # 返回waypoint对应的x,y,z的坐标
    def return_cor(self, waypoint):
        location = waypoint.transform.location
        return [location.x, location.y, location.z]

    def get_road(self, world_map):
        WAYPOINT_DISTANCE = 10
        topology = world_map.get_topology()
        road_list = []
        for wp_pair in topology:
            current_wp = wp_pair[0]
            # Check if there is a road with no previous road, this can happen
            # in opendrive. Then just continue.
            if current_wp is None:
                continue
            # First waypoint on the road that goes from wp_pair[0] to wp_pair[1].
            current_road_id = current_wp.road_id
            wps_in_single_road = [self.return_cor(current_wp)]
            # While current_wp has the same road_id (has not arrived to next road).
            while current_wp.road_id == current_road_id:
                # Check for next waypoints in aprox distance.
                available_next_wps = current_wp.next(WAYPOINT_DISTANCE)
                # If there is next waypoint/s?
                if available_next_wps:
                    # We must take the first ([0]) element because next(dist) can
                    # return multiple waypoints in intersections.
                    current_wp = available_next_wps[0]
                    wps_in_single_road.append(self.return_cor(current_wp))
                else:  # If there is no more waypoints we can stop searching for more.
                    break
            pcd1 = o3d.geometry.PointCloud()
            pdc2 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(wps_in_single_road[:-1])
            pdc2.points = o3d.utility.Vector3dVector(wps_in_single_road[1:])
            corr = [(i, i + 1) for i in range(len(wps_in_single_road) - 2)]
            lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd1, pdc2, corr)
            lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
            road_list.append(lineset)
        return road_list

    def find_and_replay(self, args):
        self.recording_rawdata = True
        # road_net = self.get_road(self.world.get_map())
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        # o3d.visualization.draw_geometries(road_net+[mesh],height=1280,width=1920)
        try:
            start_time = time.time()
            replay_time = self.start_replay(args)
            while time.time() - start_time < replay_time:
                self.world.tick()
        finally:
            print('stop replay...')
            time.sleep(2)
            self.stop_replay(args)
            pass

    def start_replay(self, args):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        self.world.apply_settings(settings)

        # set the time factor for the replayer
        self.client.set_replayer_time_factor(args.time_factor)

        # replay the session
        output = self.client.replay_file(args.recorder_filename, args.start, args.duration, args.camera)
        replay_time = self.find_replay_time(output, args.duration)
        print('start replay...{}'.format(str(output)))
        return replay_time

    def stop_replay(self, args):
        actor_list = []
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor_list.append(actor.id)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        if self.args.sync:  # and synchronous_master:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        print('destroying %d vehicles' % len(actor_list))
        self.world = self.client.reload_world()
        self.map.initial_spectator(args.spectator_point)
        exit()

    def find_replay_time(self, output, duration):
        index_start = output.index('-') + 2
        index_end = output.index('(') - 2
        total_time = float(output[index_start:index_end])
        if duration == 0:
            return total_time
        else:
            return duration


if __name__ == "__main__":

    args = Args(sys.argv)
    scenario = Scenario(args)
    if args.task == 'spawn':
        scenario.look_for_spawn_points(args)
    elif args.task == 'record':
        scenario.generate_data(args)
    elif args.task == 'replay':
        scenario.find_and_replay(args)
