#!/bin/bash

if [ $# -ne 1 ]
then
   echo "please input  world_size!"
   exit 1
fi

world_size=$1



for((i=0; i<${world_size}; i++))
do
	python dis_mnist.py --world-size ${world_size}  --rank ${i}   > progress_${i}.log 2>&1 &
done

