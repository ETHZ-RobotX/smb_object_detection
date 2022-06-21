#!/usr/bin/env bash

export ROS_IP=192.168.55.1
export ROS_MASTER_URI=http://192.168.55.100:11311
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
source /home/robotx/catkin_workspaces/smb_dev/devel/setup.bash

exec "$@"