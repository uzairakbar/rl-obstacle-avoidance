# group-d

Project of Group D

Running the application
Option 1
1.  Move the package to your "catkin_ws"
2.  run "catkin_make" in the "catkin_ws" directory .
3.  Run "source devel/setup.bash" command in the "catkin_ws" directory
4.  Run "roslaunch rl_tb_lidar tb_rl_m1.launch"
5.  To run different maps change the launch file i.e. tb_rl_m2.launch.

Option 2
1.  Move the package to your "catkin_ws"
2.  run "catkin_make" in the "catkin_ws" directory .
3.  Run "source devel/setup.bash" command in the "catkin_ws" directory
4.  Run "roslaunch rl_tb_lidar tb_stage_m1.launch" to launch only stage.
5.  Run the "turtlebot_run.py" exetuable from pycharm or another terminal e.g. "python turtlebot_run.py".
6.  To run different maps change the launch file i.e. tb_stage_m2.launch.
