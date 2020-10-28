## Build
1. Get things download
```sh
cd ~/cartorgrapher_ws/src
git clone https://github.com/ablk/cartographer.git -b develop
git clone https://github.com/ablk/cartographer_ros.git
```
2. Install dependency
```sh
cd ~/cartorgrapher_ws
src/cartographer/scripts/install_proto3.sh
src/cartographer/scripts/install_abseil.sh
```

3. Compile
```sh
catkin_make_isolated --only-pkg-with-deps cartographer_ros cartographer_rviz --install --use-ninja
```

## Addition Parameters

* TRAJECTORY_BUILDER.pure_localization : (bool) \
Modified localization mode, reuse deprecated parameter.

* TRAJECTORY_BUILDER_2D.filter_moving : (bool) \
Enable moving filter.

* TRAJECTORY_BUILDER_2D.moving_threshold : (double, range(0~1)) \
After passing ceres scan matcher, if cost of a scan point > threshold, it will be classified as moving point.

* TRAJECTORY_BUILDER_2D.sparse_threshold : (double, range(0~1)) \
In a 3x3 window, if unknown neighbor amout / 8 > threshold, then the scan point will be ignored by moving filtered, so that we can keep the points out of matching submap.

* TRAJECTORY_BUILDER_2D.dilate_radius : (double) \
If a point is near to a static point, it is seen as static as well.

## Setting
* origin cartographer mapping
```lua
TRAJECTORY_BUILDER_2D.filter_moving = false
TRAJECTORY_BUILDER.pure_localization = false
```
* origin cartographer localization
```lua
TRAJECTORY_BUILDER_2D.filter_moving = false
TRAJECTORY_BUILDER.pure_localization = false
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}
```
* mapping + filter moving
```lua
TRAJECTORY_BUILDER_2D.filter_moving = true
TRAJECTORY_BUILDER.pure_localization = false
```
* modified localization
```lua
TRAJECTORY_BUILDER.pure_localization = true
TRAJECTORY_BUILDER_2D.filter_moving = false
```

## TODO
1.Add thead for localization
2.Handle case if lost again