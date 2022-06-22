#!/usr/bin/env bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
source /home/robotx/catkin_workspaces/smb_dev/devel/setup.bash

exec "$@"