#!/bin/bash

for i in d12/snapshot_iter_*.caffemodel ; do ln -sf `pwd`/$i d12/network.caffemodel ;   ./zv --batch ~/ball_videos/dark\ purple/20160114_0.avi; echo $i; done

