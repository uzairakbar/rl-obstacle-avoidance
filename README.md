# Applied Reinforcement Learning (SS2019)
## Group D

Instructions for Stage Simulator:

1. Move the `rl_tb_lidar` and `stage_ros_u` folders to `catkin_ws/src` directory.
2. run `catkin_make` in the `catkin_ws` directory.
3. Run `source devel/setup.bash` command in the `catkin_ws` directory.
4. Run `roslaunch rl_tb_lidar tb_stage_m1.launch` to launch only stage.
5. Open an another terminal, go to the directory of the python script e.g. `cd ~/catkin_ws/src/rl_tb_lidar/src$` and run `python turtlebot_run.py map1` with a `map1` argument.
6. To run different maps change the launch file to e.g. `tb_stage_m2.launch` and use `map2` to run python command i.e. `python turtlebot_run.py map2`.

Instructions to launch real Turtlebot:


1. SSH connection: 
    * Pick one of the turtlebots located in the cabinet and power on. The turtlebot should be automatically connected to the network.
    * Open a terminal on your remote computer and write  `ssh t2@<robot_name>` e.g. `ssh t2@pik-koenig` and the password is `apfelkuchen`.
2. Once you connected to the real turtlebot, we need to learn the IP address of the real turtlebot to set as a `rosmaster`. `ifconfig` is probably not working as I tested so far, write `ip addr show` command to the ssh connected terminal. The result should be similar to e.g. `129.187.240.61`.
3. On your remote computer, you need to change the `ROS_MASTER_URI` address to assign the real turtlebot as a master. Therefore, open a new terminal on your remote computer and run the following command `export ROS_MASTER_URI=http://<IP address of TurtleBot>:11311` e.g. `export ROS_MASTER_URI=http://129.187.240.61:11311`. You have to run this command everytime when you open a new terminal.
4. Run the `env | grep ROS` to check the IP address of the ROS_MASTER. Those steps should be sufficient to reach the published topics and control the turtlebot from the remote computer but for details please check the book in the ref folder(Chapter 3 "Setting up to control a real TurtleBot" section).
5. Launching a basic LIDAR application.
    * From your ssh connected terminal write `roslaunch rplidar_ros rplidar.launch` to start rplidar node which only publishes to the `/scan` topic.
    * From your remote computer terminal, write `rosrun rplidar_ros rplidarNodeClient` or `rostopic echo /scan` to view raw scan result.
    * For the details of the `rplidar` package check [this](http://wiki.ros.org/rplidar).
6. To save the published result of the `/scan` topic, open a new terminal and go to the folder that you want to save the file. Then, write `rostopic echo /scan >>test.txt`.

7. To launch real turtlebot, connect the turtlebot with ssh.
    *  source the catkin_ws `source catkin_ws/devel/setup.bash` 
    * `roslaunch rl_tb_lidar real_turtlebot_filtered.launch`  
    * Open another terminal and connet with ssg, go to the src folder to run the script `python main_real.py config_real.yaml`  