# Applied Reinforcement Learning (SS2019)
## Group D - Obstacle Avoidance

### Results:
#### Sensor Model:
#### State Representation:
#### Linear Value Function Approximation:
#### Supplementary Material:
* Sensor Model: [Notebook](https://gitlab.ldv.ei.tum.de/arl19/group-d/blob/develop/src/rl_tb_lidar/src/utils/sensormodel/lidar_sensor_model.ipynb)
* Auto-Encoders: [Notebook](https://gitlab.ldv.ei.tum.de/arl19/group-d/blob/develop/src/rl_tb_lidar/src/utils/autoencoders/vae_experiments.ipynb)

### Instructions:

1. Move the `rl_tb_lidar` and `stage_ros_u` folders to `catkin_ws/src` directory.
2. run `catkin_make` in the `catkin_ws` directory.
3. Run `source devel/setup.bash` command in the `catkin_ws` directory.
4. Run `roslaunch rl_tb_lidar tb_stage_m1.launch` to launch only stage.
5. Open an another terminal, go to the directory of the python script e.g. `cd ~/catkin_ws/src/rl_tb_lidar/src` and run `python main.py config.yaml` with a `config.yaml` argument.
5. To try different configurations, edit the `config.yaml` file accordingly. Make sure to specify the correct `nA`, `nS` for the reinforcement learning agent (these argument will be computed automatically in the coming versions, including heavy editing of other reinforcement learning agent interface/functionality).
6. To run different maps change the launch file to e.g. `tb_stage_m2.launch` and use `'map2'` as the `map` argument in `config.yaml` (might be changed later to a command line argument for ease of use).

