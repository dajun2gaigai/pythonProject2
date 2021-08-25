import carla
import random
from pathlib import Path
import os
from queue import Queue
from queue import Empty

ROOT_PATH = Path(__file__).parent
output_path = ROOT_PATH / 'raw_data'

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if 'lidar' in sensor_name:
        sensor_data.save_to_disk(os.path.join(output_path, '%06d.ply' % sensor_data.frame))
    if 'camera' in sensor_name:
        sensor_data.save_to_disk(os.path.join(output_path, '%06d' % sensor_data.frame))
    sensor_queue.put((sensor_data.frame, sensor_name))

def main():
    global world, client
    actor_list = []
    sensor_list = []

    try:
        client = carla.Client('localhost',2000)
        client.set_timeout(2.0)

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        weather = carla.WeatherParameters(cloudiness=10.0, precipitation=10.0,fog_density=10.0)
        world.set_weather(weather)

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        #创建队列，实现同步
        sensor_queue = Queue()

        ego_vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
        ego_vehicle_bp.set_attribute('color','0,0,0')
        transform = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(ego_vehicle_bp,transform)
        ego_vehicle.set_autopilot(True)

        actor_list.append(ego_vehicle)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5,z=2.4))
        camera = world.spawn_actor(camera_bp,camera_transform,attach_to=(ego_vehicle))
        camera.listen(lambda image: sensor_callback(image,sensor_queue,'camera'))
        sensor_list.append(camera)

        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(90000))
        lidar_bp.set_attribute('rotation_frequency',str(40))
        lidar_bp.set_attribute('range',str(20))

        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        lidar = world.spawn_actor(lidar_bp,lidar_transform,attach_to=ego_vehicle)
        lidar.listen(lambda point_cloud:sensor_callback(point_cloud,sensor_queue,'lidar'))
        sensor_list.append(lidar)

#利用queue在callback中的put和world.tick()后的get实现同步
        while True:
            world.tick()
            spectator = world.get_spectator()
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location+carla.Location(z=20),carla.Rotation(pitch=-90)))

            try:
                for i in range(0,len(sensor_list)):
                    #get(block,timeout): block为True表示队列为空，则暂停线程等待，等待1s,如果没得到则抛出Empty异常，如果队列为空且为false,则抛出Empty异常。
                    s_frame = sensor_queue.get(True,1.0)
                    print("  frame:%d, sensor:%s" % (s_frame[0],s_frame[1]))
            except Empty:
                print("  some of the sensor information is missed")
    finally:
        world.apply_settings(original_settings)
        print("distroy actors")
        client.apply_batch([carla.command.DestroyActor(x)  for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()
        print('done')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' Exit by user.')
