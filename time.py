import random
import carla
import os
from pathlib import Path


ROOT_PATH = Path(__file__).parent
output_path = ROOT_PATH / 'raw_data'
actor_list = []
sensor_list = []

client = carla.Client('localhost',2000)
client.set_timeout(2.0)

world = client.get_world()

weather = carla.WeatherParameters(cloudiness=10.0, precipitation=20.0, fog_density=10.0)
world.set_weather(weather)

# 拿到这个世界所有物体的蓝图
blueprint_library = world.get_blueprint_library()
# 从浩瀚如海的蓝图中找到奔驰的蓝图
ego_vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
# 给我们的车加上特定的颜色
ego_vehicle_bp.set_attribute('color', '0, 0, 0')

# 找到所有可以作为初始点的位置并随机选择一个
transform = random.choice(world.get_map().get_spawn_points())
# 在这个位置生成汽车
ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)

actor_list.append(ego_vehicle)

# 再给它挪挪窝
# location = ego_vehicle.get_location()
# location.x += 0.5
# ego_vehicle.set_location(location)
# 把它设置成自动驾驶模式
ego_vehicle.set_autopilot(True)
# 我们可以甚至在中途将这辆车“冻住”，通过抹杀它的物理仿真
# actor.set_simulate_physics(False)

# 如果注销单个Actor
# ego_vehicle.destroy()
# 如果你有多个Actor 存在list里，想一起销毁。
# client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

#camera & Lidar
#位置都是相对汽车中心点的位置（以米计量）。
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
#我们还要对相机定义它的callback function,定义每次仿真世界里传感器数据传回来后，
# 我们要对它进行什么样的处理。在这个教程里我们只需要简单地将文件存在硬盘里。
camera.listen(lambda image: image.save_to_disk(os.path.join(output_path, '%06d.png' % image.frame)))
sensor_list.append(camera)

lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', str(32))
lidar_bp.set_attribute('points_per_second', str(90000))
lidar_bp.set_attribute('rotation_frequency', str(40))
lidar_bp.set_attribute('range', str(20))
#接着把lidar放置在奔驰上, 定义它的callback function.
lidar_location = carla.Location(0, 0, 2)
lidar_rotation = carla.Rotation(0, 0, 0)
lidar_transform = carla.Transform(lidar_location, lidar_rotation)
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
lidar.listen(lambda point_cloud: point_cloud.save_to_disk(os.path.join(output_path, '%06d.ply' % point_cloud.frame)))
sensor_list.append(lidar)

while True:
    #将spectator对准小车
    spectator = world.get_spectator()
    transform = ego_vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                        carla.Rotation(pitch=-90)))



