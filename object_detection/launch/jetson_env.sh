#!/usr/bin/env bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
source ~/smb_ws/devel/setup.bash

exec "$@"
