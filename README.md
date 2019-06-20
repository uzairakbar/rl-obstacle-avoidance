# Applied Reinforcement Learning (SS2019)
## Group D

Instructions for Stage Simulator:

1. Move the `rl_tb_lidar` and `stage_ros_u` folders to `catkin_ws/src` directory.
2. run `catkin_make` in the `catkin_ws` directory.
3. Run `source devel/setup.bash` command in the `catkin_ws` directory.
4. Run `roslaunch rl_tb_lidar tb_stage_m1.launch` to launch only stage.
5. Open an another terminal, go to the directory of the python script e.g. `cd ~/catkin_ws/src/rl_tb_lidar/src$` and run `python turtlebot_run.py map1` with a `map1` argument.
6. To run different maps change the launch file to e.g. `tb_stage_m2.launch` and use `map2` to run python command i.e. `python turtlebot_run.py map2`.
