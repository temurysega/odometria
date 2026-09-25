#!/bin/bash
# проверка пакета в ROS 2 Humble внутри docker образа ros:humble:
# сборка colcon, запуск узла, проигрывание bag, запись выходов, замеры
# запуск: docker run --rm -v REPO:/repo:ro -v DATA:/bags:ro -v OUT:/out ros:humble bash /repo/tools/ros_check.sh
source /opt/ros/humble/setup.bash
BAG=${BAG:-30618_01f73500}
SECONDS_PLAY=${SECONDS_PLAY:-300}
rm -rf /tmp/ws && mkdir -p /tmp/ws && cp -r /repo/src /tmp/ws/src
cd /tmp/ws
colcon build --symlink-install > /out/build.log 2>&1
echo "build exit $?" | tee -a /out/build.log
tail -3 /out/build.log
source install/setup.bash
rm -rf /out/rec
ros2 run odometria latency_probe --ros-args -p output_file:=/out/latency.json -p report_period:=60.0 > /out/probe.log 2>&1 &
ros2 run odometria odometry_node > /out/node.log 2>&1 &
sleep 4
NODE_PID=$(pgrep -f "lib/odometria/odometry_node" | head -1)
echo "node pid $NODE_PID"
ros2 bag record -o /out/rec /result/velocity /result/position /result/acceleration /result/diagnostics /result/slip_detected /sensing/gnss/master/fix /sensing/gnss/master/vel /sensing/gnss/rover/fix > /out/record.log 2>&1 &
REC_PID=$!
sleep 3
START=$(date +%s.%N)
CPU0=$(awk '{print $14+$15}' /proc/$NODE_PID/stat)
timeout $SECONDS_PLAY ros2 bag play /bags/$BAG > /out/play.log 2>&1
END=$(date +%s.%N)
CPU1=$(awk '{print $14+$15}' /proc/$NODE_PID/stat)
TICKS=$(getconf CLK_TCK)
grep -E "VmHWM|VmRSS|Threads" /proc/$NODE_PID/status | tee /out/memory.txt
python3 -c "print('cpu share of one core', ($CPU1-$CPU0)/$TICKS/($END-$START))" | tee -a /out/memory.txt
sleep 2
kill -INT $REC_PID
pkill -INT -f latency_probe
sleep 3
pkill -INT -f odometry_node
sleep 2
cat /out/latency.json
grep -E "запущена|выставка|Error|Traceback" /out/node.log | head -20
ros2 bag info /out/rec | grep -E "Topic|Count|Duration" | head -20
