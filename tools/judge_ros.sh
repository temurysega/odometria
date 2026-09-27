#!/bin/bash
# прогон проверочного узла организаторов (check-code, hackathon_solution_checker)
# вместе с нашим узлом в ros:humble на bag с эталоном /localization/kinematic_state
# запуск: docker run --rm -v REPO:/repo:ro -v CHECK_CODE:/check:ro -v OUT:/out ros:humble bash /repo/tools/judge_ros.sh
source /opt/ros/humble/setup.bash
BAG=${BAG:-/check/bags/30618_88aea4d9}
rm -rf /tmp/ws && mkdir -p /tmp/ws/src
cp -r /repo/src/odometria /repo/src/tram_vehicle_msgs /tmp/ws/src/
cp -r /check/src/checker_ros /tmp/ws/src/
cd /tmp/ws
colcon build > /out/judge_build.log 2>&1
echo "build exit $?" | tee -a /out/judge_build.log
source install/setup.bash
ros2 run hackathon_solution_checker metrics --ros-args -p report_period_sec:=60.0 > /out/judge_metrics.log 2>&1 &
ros2 run odometria odometry_node > /out/judge_node.log 2>&1 &
sleep 5
ros2 bag play --rate ${RATE:-1} $BAG > /out/judge_play.log 2>&1
sleep 3
pkill -INT -f odometry_node
sleep 1
# при остановке проверочный узел печатает итог
pkill -INT -f "hackathon_solution_checker/metrics"
sleep 3
grep -E "выставка|Error|Traceback" /out/judge_node.log | head -5
tail -2 /out/judge_metrics.log
