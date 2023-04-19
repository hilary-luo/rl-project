/* Lightweight ROS2 robot simulator for exploration task
 * Implements simulated laser scanning and path planning
 * Odometry is simulated to be accurate to avoid the computation for localization

 * Requirements:
 * - ROS2 (https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
 * - OpenCV 4 or higher (built from source to get the header files: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

 * Launching using terminal:
 * cd <path to folder>
 * colcon build
 * source install/setup.bash
 * ros2 launch robot_sim robot_sim.launch.py

 * Testing using a new terminal:
 * cd <path to folder>
 * source install/setup.bash
 * ros2 service call /load_map robot_sim/srv/LoadMap "{map_path: 'map.bmp', threshold: 200, resolution: 0.05}"
 * ros2 service call /set_robot_pose robot_sim/srv/SetRobotPose "{pose: {position: {x: 0, y: 0}}}"
 * ros2 service call /compute_path robot_sim/srv/ComputePath "{pose: {position: {x: 5, y: 4}}}"
 * ros2 action send_goal --feedback /navigate_to_pose robot_sim/action/NavigateToPose "{speed: 0.5, pose: {position: {x: 5, y: 4}}}"
 * ros2 service call /reset_map robot_sim/srv/ResetMap "{keep_robot: true}"
 * ros2 service call /reset_map robot_sim/srv/ResetMap "{keep_robot: false}"
*/

#include <string>
#include <cmath>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "robot_sim/srv/load_map.hpp"
#include "robot_sim/srv/reset_map.hpp"
#include "robot_sim/srv/set_robot_pose.hpp"
#include "robot_sim/srv/compute_path.hpp"
#include "robot_sim/action/navigate_to_pose.hpp"

/* Map is defined by a 2D array where 0 = free/unoccupied, 100 = occupied, -1 = unknown

 * When loading a map image, it is converted to grayscale (0-255) and thresholded as follows:
 * if pixel >= threshold, it is labelled as free (light colors / white)
 * if pixel < threshold, it is labelled as occupied (dark colors / black)
 * There are no unknown cells in a loaded full map, only in the current map being explored

 * Costmap for navigation is defined by a 2D array where 0 = free/safe, 100 = occupied/avoid
 * The costmap is computed by "growing" the occupied areas by the radius of the robot to ensure the entire robot is in free space

 * A current map is also maintained which tracks the area explored by the robot
 * Navigation will only travel to areas that are known to be free in the currently explored map, and the entire path will be only along free cells
*/

#define ROBOT_RADIUS     0.2 // meters, radius of the simulated robot, used to keep distance from obstacles
#define LASER_RANGE      12  // meters, simulated range of laser scanner
#define LASER_SCAN_LINES 1000 // number of simulated laser scan lines covering 360 degrees
#define PUBLISH_PERIOD   100 // milliseconds, how often data is published
#define PATH_TIMEOUT     1000 // milliseconds, timeout for path computation
#define MAP_MIN_SIZE     100 // cells for height and width, minimum size of maps
#define PATH_STRIDE_MAX  5   // maximum number of path cells to stride in 1 iteration, limits navigation speed

class RobotSimServer : public rclcpp::Node
{
public:
    using TransformBroadcaster = tf2_ros::TransformBroadcaster;
    using Pose = geometry_msgs::msg::Pose;
    using PoseStamped = geometry_msgs::msg::PoseStamped;
    using PointStamped = geometry_msgs::msg::PointStamped;
    using TransformStamped = geometry_msgs::msg::TransformStamped;
    using OccupancyGrid = nav_msgs::msg::OccupancyGrid;
    using Path = nav_msgs::msg::Path;
    using LoadMap = robot_sim::srv::LoadMap;
    using ResetMap = robot_sim::srv::ResetMap;
    using SetRobotPose = robot_sim::srv::SetRobotPose;
    using ComputePath = robot_sim::srv::ComputePath;
    using NavigateToPose = robot_sim::action::NavigateToPose;
    using NavigateToPoseGoal = rclcpp_action::ServerGoalHandle<NavigateToPose>;


    // Constructor
    RobotSimServer() : rclcpp::Node("robot_sim_server")
    {
        // Create the publishers
        pub_transform = std::make_unique<TransformBroadcaster>(*this);
        pub_map_full = this->create_publisher<OccupancyGrid>("map_full", 1); // Full map
        pub_costmap_full = this->create_publisher<OccupancyGrid>("costmap_full", 1); // Full costmap
        pub_map = this->create_publisher<OccupancyGrid>("map", 1); // Explored map
        pub_costmap = this->create_publisher<OccupancyGrid>("costmap", 1); // Explored costmap
        pub_path = this->create_publisher<Path>("path", 1); // Path robot is navigating using navigate_to_pose action
        pub_path_computed = this->create_publisher<Path>("path_computed", 1); // Path computed using compute_path service
        pub_pose = this->create_publisher<PoseStamped>("pose", 1); // Robot current pose
        pub_robot = this->create_publisher<PointStamped>("robot", 1); // Robot current pose in point form for RViz display
        
        // Initialize the services and actions
        srv_load_map = this->create_service<LoadMap>("load_map",
            std::bind(&RobotSimServer::load_map_callback, this, std::placeholders::_1, std::placeholders::_2));
        srv_reset_map = this->create_service<ResetMap>("reset_map",
            std::bind(&RobotSimServer::reset_map_callback, this, std::placeholders::_1, std::placeholders::_2));
        srv_set_robot_pose = this->create_service<SetRobotPose>("set_robot_pose",
            std::bind(&RobotSimServer::set_robot_pose_callback, this, std::placeholders::_1, std::placeholders::_2));
        srv_compute_path = this->create_service<ComputePath>("compute_path",
            std::bind(&RobotSimServer::compute_path_callback, this, std::placeholders::_1, std::placeholders::_2));
        action_nav = rclcpp_action::create_server<NavigateToPose>(this, "navigate_to_pose",
            std::bind(&RobotSimServer::navigate_to_pose_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&RobotSimServer::navigate_to_pose_cancel, this, std::placeholders::_1),
            std::bind(&RobotSimServer::navigate_to_pose_accepted, this, std::placeholders::_1));
        timer_publish = this->create_wall_timer(std::chrono::milliseconds(PUBLISH_PERIOD),
            std::bind(&RobotSimServer::publish_callback, this));

        // Initialize the maps to blank
        map_full.create(MAP_MIN_SIZE, MAP_MIN_SIZE, CV_8SC1);
        map_full = cv::Scalar::all(100);
        costmap_full.create(MAP_MIN_SIZE, MAP_MIN_SIZE, CV_8SC1);
        costmap_full = cv::Scalar::all(100);
        map_current.create(MAP_MIN_SIZE, MAP_MIN_SIZE, CV_8SC1);
        map_current = cv::Scalar::all(-1);
        costmap_current.create(MAP_MIN_SIZE, MAP_MIN_SIZE, CV_8SC1);
        costmap_current = cv::Scalar::all(-1);

        path.clear();
        path_computed.clear();

        RCLCPP_INFO(this->get_logger(), "Initialized robot simulator server");
    }

private:

    // Publishers, services and actions
    std::unique_ptr<tf2_ros::TransformBroadcaster> pub_transform;
    rclcpp::Publisher<OccupancyGrid>::SharedPtr pub_map_full;
    rclcpp::Publisher<OccupancyGrid>::SharedPtr pub_costmap_full;
    rclcpp::Publisher<OccupancyGrid>::SharedPtr pub_map;
    rclcpp::Publisher<OccupancyGrid>::SharedPtr pub_costmap;
    rclcpp::Publisher<Path>::SharedPtr pub_path;
    rclcpp::Publisher<Path>::SharedPtr pub_path_computed;
    rclcpp::Publisher<PoseStamped>::SharedPtr pub_pose;
    rclcpp::Publisher<PointStamped>::SharedPtr pub_robot;
    rclcpp::Service<LoadMap>::SharedPtr srv_load_map;
    rclcpp::Service<ResetMap>::SharedPtr srv_reset_map;
    rclcpp::Service<SetRobotPose>::SharedPtr srv_set_robot_pose;
    rclcpp::Service<ComputePath>::SharedPtr srv_compute_path;
    rclcpp_action::Server<NavigateToPose>::SharedPtr action_nav;    
    rclcpp::TimerBase::SharedPtr timer_publish;
    
    // Data members
    int map_width = MAP_MIN_SIZE, map_height = MAP_MIN_SIZE;
    float map_resolution = 0.1;
    int robot_x = -1, robot_y = -1;
    cv::Mat map_full, costmap_full, map_current, costmap_current;
    std::vector<cv::Point> path;
    std::vector<cv::Point> path_computed;
    std::mutex data_mutex;
    rclcpp_action::GoalUUID active_goal;
    

    // Callback to publish data to all topics
    void publish_callback()
    {
        const std::lock_guard<std::mutex> lock(data_mutex);

        // Find a bounding box to crop the explored map and costmap
        // This ensures that the published maps are the minimum size, and grow as more area is explored
        // This is how TurtleBot simulation behaves and has been repicated

        cv::Mat map_mask = (map_current >= 0);
        std::vector<cv::Point> map_coords;
        cv::findNonZero(map_mask, map_coords); // List all explored pixels

        cv::Rect crop(map_width/2, map_height/2, 0, 0);
        if (map_coords.size() != 0) // Current explored map is not empty
           crop = cv::boundingRect(map_coords); // Find minimum spanning bounding box
        
        // Make sure the map has a minimum starting size
        if (crop.width < MAP_MIN_SIZE)
            crop = cv::Rect(std::max(0, crop.x - (MAP_MIN_SIZE - crop.width)/2), crop.y, MAP_MIN_SIZE, crop.height);
        if (crop.height < MAP_MIN_SIZE)
            crop = cv::Rect(crop.x, std::max(0, crop.y - (MAP_MIN_SIZE - crop.height)/2), crop.width, MAP_MIN_SIZE);

        // Robot location in map coordinates in meters
        // The map coordinate frame origin is at the center of the full map
        float robot_x_map = (robot_x - map_width/2) * map_resolution;
        float robot_y_map = (robot_y - map_height/2) * map_resolution;
        
        rclcpp::Time time_msg = this->get_clock()->now();

        TransformStamped tf;
        tf.header.stamp = time_msg;
        tf.header.frame_id = "world";
        tf.child_frame_id = "map";
        tf.transform.translation.x = tf.transform.translation.y = tf.transform.translation.z = 0.0;
        tf.transform.rotation.x = tf.transform.rotation.y = tf.transform.rotation.z = 0.0;
        tf.transform.rotation.w = 1.0;
                
        PointStamped point_msg;
        point_msg.header.frame_id = "map";
        point_msg.header.stamp = time_msg;
        point_msg.point.x = robot_x_map;
        point_msg.point.y = robot_y_map; 
        point_msg.point.z = 0.0;
        
        PoseStamped pose_msg;
        pose_msg.header.frame_id = "map";
        pose_msg.header.stamp = time_msg;
        pose_msg.pose.position.x = robot_x_map;
        pose_msg.pose.position.y = robot_y_map;
        pose_msg.pose.position.z = 0.0;
        pose_msg.pose.orientation.x = pose_msg.pose.orientation.y = pose_msg.pose.orientation.z = 0.0;
        pose_msg.pose.orientation.w = 1.0;      

        Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = time_msg;
        for (unsigned int i=0; i < path.size(); i++)
        {
            float path_x_map = (path[i].x - map_width/2) * map_resolution;
            float path_y_map = (path[i].y - map_height/2) * map_resolution;

            PoseStamped path_pose_msg;
            path_pose_msg.header.frame_id = "map";
            path_pose_msg.header.stamp = time_msg;
            path_pose_msg.pose.position.x = path_x_map;
            path_pose_msg.pose.position.y = path_y_map;
            path_pose_msg.pose.position.z = 0.0;
            path_pose_msg.pose.orientation.x = path_pose_msg.pose.orientation.y = path_pose_msg.pose.orientation.z = 0.0;
            path_pose_msg.pose.orientation.w = 1.0;
            path_msg.poses.push_back(path_pose_msg);
        }

        Path path_computed_msg;
        path_computed_msg.header.frame_id = "map";
        path_computed_msg.header.stamp = time_msg;
        for (unsigned int i=0; i < path_computed.size(); i++)
        {
            float path_x_map = (path_computed[i].x - map_width/2) * map_resolution;
            float path_y_map = (path_computed[i].y - map_height/2) * map_resolution;

            PoseStamped path_pose_msg;
            path_pose_msg.header.frame_id = "map";
            path_pose_msg.header.stamp = time_msg;
            path_pose_msg.pose.position.x = path_x_map;
            path_pose_msg.pose.position.y = path_y_map;
            path_pose_msg.pose.position.z = 0.0;
            path_pose_msg.pose.orientation.x = path_pose_msg.pose.orientation.y = path_pose_msg.pose.orientation.z = 0.0;
            path_pose_msg.pose.orientation.w = 1.0;
            path_computed_msg.poses.push_back(path_pose_msg);
        }
        
        // Publish all of the computed messages
        pub_transform->sendTransform(tf);
        pub_robot->publish(point_msg);
        pub_pose->publish(pose_msg);
        pub_path->publish(path_msg);
        pub_path_computed->publish(path_computed_msg);
        pub_map_full->publish(map_data_to_msg(map_full, time_msg, map_width/2, map_height/2));
        pub_costmap_full->publish(map_data_to_msg(costmap_full, time_msg, map_width/2, map_height/2));
        pub_map->publish(map_data_to_msg(map_current(crop), time_msg, map_width/2 - crop.x, map_height/2 - crop.y));
        pub_costmap->publish(map_data_to_msg(costmap_current(crop), time_msg, map_width/2 - crop.x, map_height/2 - crop.y));
    }


    // Convert cv::Mat map to Occupancy grid with given offset for origin
    OccupancyGrid map_data_to_msg(const cv::Mat &map_matrix, const rclcpp::Time &time_msg, int x_offset_origin, int y_offset_origin) 
    {        
        Pose origin; // location of 0,0 corner of map image from the origin in the map coordinate frame
        origin.position.x = - x_offset_origin * map_resolution;
        origin.position.y = - y_offset_origin * map_resolution;
        origin.position.z = 0.0;
        origin.orientation.x = origin.orientation.y = origin.orientation.z = 0.0;
        origin.orientation.w = 1.0;
        
        OccupancyGrid map_msg;
        map_msg.header.frame_id = "map";
        map_msg.header.stamp = time_msg;
        map_msg.info.map_load_time = time_msg;
        map_msg.info.resolution = map_resolution;
        map_msg.info.width = map_matrix.cols;
        map_msg.info.height = map_matrix.rows;
        map_msg.info.origin = origin;        
        if (map_matrix.isContinuous()) 
            map_msg.data.assign(map_matrix.data, map_matrix.data + map_matrix.total());
        else 
            for (int i = 0; i < map_matrix.rows; ++i) 
                map_msg.data.insert(map_msg.data.end(), map_matrix.ptr<char>(i), map_matrix.ptr<char>(i) + map_matrix.cols);
        
        return map_msg;
    }


    void load_map_callback(const std::shared_ptr<LoadMap::Request> request, std::shared_ptr<LoadMap::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Load map service called");
        const std::lock_guard<std::mutex> lock(data_mutex);
        response->success = load_map(request->map_path, request->threshold, request->resolution, request->flip, request->rotate);
        if (response -> success) 
            response->map = map_data_to_msg(map_full, this->get_clock()->now(), map_width/2, map_height/2);
        RCLCPP_INFO(this->get_logger(), "Load map service completed (success: {%s})", response->success ? "true" : "false");
    }

    
    // Load a map from a provided image path, divided into free vs occupied based on given intensity threshold
    // Resolution provides scaling in meters/pixel, and optionally horizontally flip and rotate (multiples of 90 deg clockwise) the image
    bool load_map(std::string map_path, unsigned int threshold, float resolution, bool flip = false, unsigned int rotate = 0)
    {
        if (resolution <= 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Map resolution invalid, must be greater than zero");
            return false;
        }
        
        cv::Mat img = cv::imread(map_path, cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Error opening map image");
            return false;
        }
        
        if (img.rows < MAP_MIN_SIZE or img.cols < MAP_MIN_SIZE)
        {
            RCLCPP_ERROR(this->get_logger(), "Map too small, must be at least 100 x 100 cells");
            return false;
        }
        
        cv::Mat tmp_img;
        img.copyTo(tmp_img);
        if (flip)
            cv::flip(tmp_img, img, 1);
            
        img.copyTo(tmp_img);
        rotate = rotate % 4;
        if (rotate == 1)
            cv::rotate(tmp_img, img, cv::ROTATE_90_CLOCKWISE);
        else if (rotate == 2)
            cv::rotate(tmp_img, img, cv::ROTATE_180);
        else if (rotate == 3)
            cv::rotate(tmp_img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
                
        RCLCPP_INFO(this->get_logger(), "Loaded map image of dimension %d x %d", img.cols, img.rows);

        cv::Mat map_thresh = (img < threshold); // less than threshold becomes 100 (occupied) and rest is free (0)
        cv::Mat map_temp = (map_thresh / 255) * 100;
        map_temp.convertTo(map_full, CV_8SC1);
                
        map_width = img.cols;
        map_height = img.rows;
        map_resolution = resolution;
        
        // Compute costmap by expanding the obstacles (occupied regions) by robot radius
        int dilate_kernel_size = ceil(ROBOT_RADIUS / resolution); // Ceil to avoid robot touching the obstacles/walls
        dilate_kernel_size = dilate_kernel_size * 2 + 1; // Kernel size must be odd
        cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_kernel_size, dilate_kernel_size));
        cv::Mat map_dilated;
        cv::dilate(map_thresh, map_dilated, dilate_kernel);
        cv::Mat costmap_temp = (map_dilated / 255) * 100;
        costmap_temp.convertTo(costmap_full, CV_8SC1);
        
        // Initialize the current map and costmap to unknown
        map_current.create(img.size(), CV_8SC1);
        map_current = cv::Scalar::all(-1);
        costmap_current.create(img.size(), CV_8SC1);
        costmap_current = cv::Scalar::all(-1);
        robot_x = -1;
        robot_y = -1;
        path.clear();
        path_computed.clear();
        
        RCLCPP_INFO(this->get_logger(), "Computed occupancy and cost maps");
        return true;
    }

    
    void reset_map_callback(const std::shared_ptr<ResetMap::Request> request, std::shared_ptr<ResetMap::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Reset map service called");
        const std::lock_guard<std::mutex> lock(data_mutex);
        // Clear the explored map and costmap
        map_current = cv::Scalar::all(-1);
        costmap_current = cv::Scalar::all(-1);
        path.clear();
        path_computed.clear();     
        if (!request->keep_robot)
            robot_x = robot_y = -1;
        else
            laser_scan_update_map(robot_x, robot_y, LASER_SCAN_LINES * 3);
        response->success = true;
        RCLCPP_INFO(this->get_logger(), "Reset map service completed (success: {%s})", response->success ? "true" : "false");
    }


    void set_robot_pose_callback(const std::shared_ptr<SetRobotPose::Request> request, std::shared_ptr<SetRobotPose::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Set robot pose service called");
        const std::lock_guard<std::mutex> lock(data_mutex);

        // Convert pose to map pixel coordinates
        int x = (int)round(request->pose.position.x / map_resolution) + map_width/2;
        int y = (int)round(request->pose.position.y / map_resolution) + map_height/2;
        
        if (x < 0 || x >= map_width || y < 0 || y >= map_height)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in set_robot_pose: location is outside map bounds");
            response->success = false;
        }
        else if (costmap_full.at<char>(y, x) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in set_robot_pose: location is not in free space");
            response->success = false;
        }
        else
        {
            path.clear();
            path_computed.clear();
            robot_x = x;
            robot_y = y;
            laser_scan_update_map(x, y, LASER_SCAN_LINES * 3);
            response->success = true;
        }
        RCLCPP_INFO(this->get_logger(), "Set robot pose service completed (success: {%s})", response->success ? "true" : "false");
    }


    void compute_path_callback(const std::shared_ptr<ComputePath::Request> request, std::shared_ptr<ComputePath::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Compute path service called");
        
        path_computed.clear();
        int target_x = (int)round(request->pose.position.x / map_resolution) + map_width/2;
        int target_y = (int)round(request->pose.position.y / map_resolution) + map_height/2;

        if (robot_x < 0 || robot_x >= map_width || robot_y < 0 || robot_y >= map_height)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: robot location has not been set");
            response->success = false;
        }
        else if (target_x < 0 || target_x >= map_width || target_y < 0 || target_y >= map_height)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: destination is outside map bounds");
            response->success = false;
        }
        else if (costmap_current.at<char>(target_y, target_x) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: destination is not in free space");
            response->success = false;
        }
        else
        {
            path_computed = compute_path(robot_x, robot_y, target_x, target_y);
            if (path_computed.size() == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Error in compute_path: path computation failed");
                response->success = false;
            }
            else
            {
                rclcpp::Time time_msg = this->get_clock()->now();
                response->path.header.frame_id = "map";
                response->path.header.stamp = time_msg;
                for (unsigned int i=0; i < path_computed.size(); i++)
                {
                    float path_x_map = (path_computed[i].x - map_width/2) * map_resolution;
                    float path_y_map = (path_computed[i].y - map_height/2) * map_resolution;

                    PoseStamped path_pose_msg;
                    path_pose_msg.header.frame_id = "map";
                    path_pose_msg.header.stamp = time_msg;
                    path_pose_msg.pose.position.x = path_x_map;
                    path_pose_msg.pose.position.y = path_y_map;
                    path_pose_msg.pose.position.z = 0.0;
                    path_pose_msg.pose.orientation.x = path_pose_msg.pose.orientation.y = path_pose_msg.pose.orientation.z = 0.0;
                    path_pose_msg.pose.orientation.w = 1.0;
                    response->path.poses.push_back(path_pose_msg);
                }
                response->success = true;
                pub_path_computed->publish(response->path); // Publish the computed path for RViz visualization
            }
        }
        RCLCPP_INFO(this->get_logger(), "Compute path service completed (success: {%s})", response->success ? "true" : "false");
    }


    rclcpp_action::GoalResponse navigate_to_pose_goal(const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const NavigateToPose::Goal> goal)
    {
        (void)goal; // to avoid unused parameter warnings
        active_goal = uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }


    rclcpp_action::CancelResponse navigate_to_pose_cancel(const std::shared_ptr<NavigateToPoseGoal> goal_handle)
    {
        (void)goal_handle; // to avoid unused parameter warnings
        return rclcpp_action::CancelResponse::ACCEPT;
    }


    void navigate_to_pose_accepted(const std::shared_ptr<NavigateToPoseGoal> goal_handle)
    {
        // launch action in a new thread to avoid blocking callbacks
        std::thread{std::bind(&RobotSimServer::navigate_to_pose_execute, this, std::placeholders::_1), goal_handle}.detach();
    }


    void navigate_to_pose_execute(const std::shared_ptr<NavigateToPoseGoal> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Navigate to pose action called");
        auto response = std::make_shared<NavigateToPose::Result>();
        // Convert pose to map pixel coordinates
        int target_x = round(goal_handle->get_goal()->pose.position.x / map_resolution) + map_width/2;
        int target_y = round(goal_handle->get_goal()->pose.position.y / map_resolution) + map_height/2;
        response->success = navigate_to_pose(target_x, target_y, goal_handle->get_goal()->speed, goal_handle);
        publish_callback(); // send the updated data right away
        if(rclcpp::ok() && response->success) 
            goal_handle->succeed(response);
        else 
            goal_handle->abort(response);
        RCLCPP_INFO(this->get_logger(), "Navigate to pose action completed (success: {%s})", response->success ? "true" : "false");
    }


    bool navigate_to_pose(int target_x, int target_y, float speed, const std::shared_ptr<NavigateToPoseGoal> goal_handle = NULL)
    {
        int path_len = 1;
        float x_map, y_map;
        {
            const std::lock_guard<std::mutex> lock(data_mutex);
            path.clear();
            path_computed.clear();
            if (speed <= 0) 
            {
                RCLCPP_ERROR(this->get_logger(), "Error in navigate_to_pose: invalid speed, must be greater than zero");
                return false;
            }
            if (robot_x < 0 || robot_x >= map_width || robot_y < 0 || robot_y >= map_height)
            {
                RCLCPP_ERROR(this->get_logger(), "Error in navigate_to_pose: robot location has not been set");
                return false;
            }
            if (target_x < 0 || target_x >= map_width || target_y < 0 || target_y >= map_height)
            {
                RCLCPP_ERROR(this->get_logger(), "Error in navigate_to_pose: destination is outside map bounds");
                return false;
            }
            if (costmap_current.at<char>(target_y, target_x) != 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Error in navigate_to_pose: destination is not in free space");
                return false;
            }
            path = compute_path(robot_x, robot_y, target_x, target_y);
            if (path.size() == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Error in navigate_to_pose: path computation failed");
                return false;
            }
            path_len = (int)path.size(); // number of pixels
        }        
        float path_time = path_len * map_resolution / speed; // seconds
        float step_time = path_time / path_len; // target seconds/step
        int stride = 1, step = 1;
        auto feedback = std::make_shared<NavigateToPose::Feedback>();
        
        RCLCPP_INFO(this->get_logger(), "Starting navigation");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto pub_time = start_time;
        
        while (step < path_len)
        {
            {
                const std::lock_guard<std::mutex> lock(data_mutex);
                if (!rclcpp::ok())
                {
                    RCLCPP_ERROR(this->get_logger(), "ROS shutdown");
                    return false;
                }
                if (goal_handle != NULL && (!goal_handle->is_active() || goal_handle->is_canceling()))
                {
                    RCLCPP_ERROR(this->get_logger(), "Navigation cancelled");
                    return false;
                }
                if (goal_handle != NULL && !(goal_handle->get_goal_id() == active_goal)) // Goal was changed
                {
                    RCLCPP_ERROR(this->get_logger(), "Navigation aborted due to goal change");
                    return false;
                }
                if (robot_x < 0 || robot_y < 0 || (int)path.size() != path_len) // Robot position was reset
                {
                    RCLCPP_ERROR(this->get_logger(), "Navigation aborted due to robot/map reset");
                    return false;
                }
                                
                robot_x = path.at(step).x;
                robot_y = path.at(step).y;
                x_map = (robot_x - map_width/2) * map_resolution;
                y_map = (robot_y - map_height/2) * map_resolution;
                laser_scan_update_map(robot_x, robot_y);
            }
            
            // RCLCPP_INFO(this->get_logger(), "Step %d of %d with stride %d", step, path_len, stride); // debugging output
            
            auto curr_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = curr_time - start_time; // time in seconds

            Pose pose_msg;
            pose_msg.position.x = x_map;
            pose_msg.position.y = y_map;
            pose_msg.position.z = 0.0;
            pose_msg.orientation.x = pose_msg.orientation.y = pose_msg.orientation.z = 0.0;
            pose_msg.orientation.w = 1.0;
            feedback->current_pose = pose_msg;
            feedback->navigation_time = elapsed.count();
            feedback->distance_remaining = (path_len - step - 1) * map_resolution;
            feedback->estimated_time_remaining = feedback->distance_remaining / speed;

            std::chrono::duration<double, std::milli> pub_period = curr_time - pub_time;
            if (pub_period.count() > PUBLISH_PERIOD)
            {
                publish_callback(); // publish updated data
                if (goal_handle != NULL)
                    goal_handle->publish_feedback(feedback);
                pub_time = curr_time;
            }
            
            // Update the stride to simulate requested speed
            if (step < (path_len-1))
            {
                if (step * step_time > feedback->navigation_time)
                    stride -= 1; // slow down if we are faster than desired
                else
                    stride += 2; // speed up if we are slower
                stride = std::min(std::max(stride, 0), PATH_STRIDE_MAX);
                step += stride;
                step = std::min(step, path_len-1);
            }
            else
                step += 1;
        }
        return true;
    }


    // Used by the pathfinding algorithm, a custom optimized priority queue using doubly-linked list
    // This makes path computation about 10 times faster than using std::priority_queue
    class NodeQueue 
    {
        public:

        struct Node
        {
            int x, y; // Node coordinates on the map
            int G; // Total distance already travelled to reach the node   
            int F; // F = G + remaining distance estimate, smaller F gets priority
            Node *prev, *next; // Stores the previous and next nodes for priority queue doubly-linked list
        };

        Node *top, *end;

        NodeQueue() : top(NULL) {}

        ~NodeQueue()
        {
            if (top != NULL)
                while (top->next != NULL)
                {
                    top = top->next;
                    delete top->prev;
                }
            delete top;
        }

        void push (int x, int y, int G, int F) // Add a new node at sorted location
        {
            Node *new_node = new Node{x, y, G, F, NULL, NULL};
            if (top == NULL) // New node is the first node
            {
                top = new_node;
                end = new_node;
                return;
            }
            if (F > end->F) // New node becomes the end
            {
                end->next = new_node;
                new_node->prev = end;
                end = new_node;
                return;
            }
            Node *spot = top;
            while (F > spot->F) // Can only perform a linear search on a linked list
                spot = spot->next; // Minimize instructions running at O(n) complexity
            new_node->next = spot;
            new_node->prev = spot->prev;
            spot->prev = new_node;
            if (new_node->prev == NULL) // New node becomes the start
                top = new_node;
            else  // New node inserts before spot
                new_node->prev->next = new_node;
        }

        void pop() // Remove top node
        {
            if (top == NULL) // No nodes
                return;
            if (top->next == NULL) // Single node
            {
                delete top;
                top = NULL;
                end = NULL;
                return;
            }
            top = top->next;
            delete top->prev;
            top->prev = NULL;
        }

        void remove(int x, int y) // Remove node matching given coordinates
        {
            if (top == NULL) // No nodes
                return;
            Node *spot = top;
            while (spot->x != x || spot->y != y) // Can only perform a linear search on a linked list
            {
                spot = spot->next; // Minimize instructions running at O(n) complexity
                if (spot == NULL) // Did not find a matching node
                    return;
            }
            if (spot->prev == NULL && spot->next == NULL) // It is the only node
            {
                top = NULL;
                end = NULL;
            }
            else if (spot->prev == NULL) // It is the top node
            {
                top = spot->next;
                top->prev = NULL;
            }
            else if (spot->next == NULL) // It is the last node
            {
                end = spot->prev;
                end->next = NULL;
            }
            else // It is somewhere in between
            {
                spot->prev->next = spot->next;
                spot->next->prev = spot->prev;
            }
            delete spot;
        }
    };


    // Compute path between given coordinates through the free area in the current costmap using A* search
    std::vector<cv::Point> compute_path(int x0, int y0, int x1, int y1)				
    {
        std::vector<cv::Point> final_path;

        if (x0 < 0 || x0 >= map_width || y0 < 0 || y0 >= map_height) 
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: start location is outside map bounds");
            return final_path;
        }
        if (x1 < 0 || x1 >= map_width || y1 < 0 || y1 >= map_height)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: goal location is outside map bounds");
            return final_path;            
        }
        if (costmap_current.at<char>(y0, x0) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: start location is not in free space");
            return final_path;
        }
        if (costmap_current.at<char>(y1, x1) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute_path: goal location is not in free space");
            return final_path;
        }

        RCLCPP_INFO(this->get_logger(), "Computing path, this can take some time");

        auto start_time = std::chrono::high_resolution_clock::now();

        const int NDIR = 8; // number of possible directions to go at any position
        const int xDir[NDIR] = {1, 1, 0, -1, -1, -1, 0, 1};
        const int yDir[NDIR] = {0, 1, 1, 1, 0, -1, -1, -1};

        int runs = 0, dir, xCurr, yCurr, GCurr, xNext, yNext, GNext, FNext;
        NodeQueue queue; // priority queue of open (not-yet-checked-out) nodes

        cv::Mat map_dir(map_height, map_width, CV_8SC1, cv::Scalar::all(-1)); // map of directions to parent node
        cv::Mat map_closed(map_height, map_width, CV_8SC1, cv::Scalar::all(0)); // map of closed (checked-out) nodes
        cv::Mat map_open(map_height, map_width, CV_32SC1, cv::Scalar::all(0)); // map of open (not-yet-checked-out) nodes with F values

        queue.push(x0, y0, 0, (abs(x1 - x0) + abs(y1 - y0)) * 10); // create the start node and push into list of open nodes
    
        // A* search
        while(queue.top != NULL) 
        {
            runs++; // track number of iterations
            xCurr = queue.top->x; // get the current node with the lowest F from the list of open nodes
            yCurr = queue.top->y;
            GCurr = queue.top->G;
            queue.pop(); // remove the node from the open list

            map_open.at<int>(yCurr, xCurr) = 0;
            map_closed.at<char>(yCurr, xCurr) = 1; // mark it on the closed nodes list
            
            if(xCurr == x1 && yCurr == y1) // stop searching when the goal state is reached
            {
                std::vector<cv::Point> rev_path;
                while(!(xCurr == x0 && yCurr == y0)) // generate the path from finish to start from map_dir
                {
                    rev_path.push_back(cv::Point(xCurr, yCurr));
                    dir = map_dir.at<char>(yCurr, xCurr);
                    xCurr += xDir[dir];
                    yCurr += yDir[dir];
                }
                final_path.reserve(rev_path.size() + 1);
                final_path.push_back(cv::Point(x0, y0)); // add start point
                for(int i = rev_path.size()-1; i >= 0; i-- )
                    final_path.push_back(rev_path.at(i));

                break; // end loop
            }
            
            for(dir = 0; dir < NDIR; dir++) // generate moves in all possible directions
            {
                xNext = xCurr + xDir[dir];
                yNext = yCurr + yDir[dir];

                // skip if outside bounds or wall (obstacle) or in the closed list
                if(xNext < 0 || xNext >= map_width || yNext < 0 || yNext >= map_height || 
                    costmap_current.at<char>(yNext, xNext) != 0 || map_closed.at<char>(yNext, xNext) != 0) 
                    continue;
                                
                // calculate G and F for a child node
                GNext = GCurr + (dir % 2 == 0 ? 10 : 14); // even dir are cardinal (10), odd dir are diagonal (14 = sqrt(2)*10)
                FNext = GNext + (abs(x1 - xNext) + abs(y1 - yNext)) * 10; // Manhattan distance

                // if it is not in the open list, then add into that
                if(map_open.at<int>(yNext, xNext) == 0) 
                {
                    map_open.at<int>(yNext, xNext) = FNext;
                    map_dir.at<char>(yNext, xNext) = (dir + NDIR/2) % NDIR; // mark parent node direction (reverse)
                    queue.push(xNext, yNext, GNext, FNext);
                }
                // if already in the open list, but found better path with lower F, update it
                else if(map_open.at<int>(yNext, xNext) > FNext) 
                {    
                    map_open.at<int>(yNext, xNext) = FNext; // update the F info
                    map_dir.at<char>(yNext, xNext) = (dir + NDIR/2) % NDIR; // mark parent node direction (reverse)
                    queue.remove(xNext, yNext); // remove the unwanted node
                    queue.push(xNext, yNext, GNext, FNext);
                }
            }

            auto curr_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = curr_time - start_time;
            if (duration.count() > PATH_TIMEOUT)
            {
                RCLCPP_ERROR(this->get_logger(), "Path computation timed out");
                break;
            }
        }

        auto stop_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop_time - start_time;

        if (final_path.size() == 0)
            RCLCPP_ERROR(this->get_logger(), "Failed to compute path in %d runs that took %f milliseconds", runs, duration.count());
        else
            RCLCPP_INFO(this->get_logger(), "Computed path of length %d in %d runs that took %f milliseconds", (int)final_path.size(), runs, duration.count());

        return final_path;
    }
    

    // Simulate laser scannning to explore the map with the robot at the given coordinates
    bool laser_scan_update_map(int x, int y, int scan_lines_num = LASER_SCAN_LINES)
    {   
        if (x < 0 or x >= map_width or y < 0 or y >= map_height)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in laser_scan_update_map: location is outside map bounds");
            return false;
        }
        if (map_full.at<char>(y, x) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in laser_scan_update_map: location is not in free space");
            return false;
        }
        
        float laser_max = LASER_RANGE / map_resolution; // Max number of map cells laser can reach
        float angle_step = 2 * 3.14159265 / scan_lines_num;
        
        for (int step = 0; step < scan_lines_num; step++)
        {
            float angle = step * angle_step;
            int dx = (int)round(cos(angle) * laser_max);
            int dy = (int)round(sin(angle) * laser_max);
            int dx_abs = abs(dx);
            int dy_abs = abs(dy);
            int adder_x = dx >= 0 ? 1 : -1;
            int adder_y = dy >= 0 ? 1 : -1;
                        
            if (dx_abs > dy_abs) // Angle of ray is closer to x axis
            {
                int d = dy_abs;
                int yi = y;
                int x_occ = x;
                for (int xi = x; xi != (x+dx); xi += adder_x)
                {
                    // If the pixel is outside of the map then move on to the next ray
                    if (xi >= map_width || yi >= map_height || xi < 0 || yi < 0)
                        break;
                    // If found an occupied pixel, save endpoint, mark on current map and terminate ray
                    if (map_full.at<char>(yi, xi) == 100)
                    {
                        x_occ = xi;
                        map_current.at<char>(yi, xi) = 100;
                        break;
                    }
                    if (d > 0)
                    {
                        yi += adder_y;
                        d -= dx_abs;
                    }
                    d += dy_abs;
                }
                // If ray intersected an occupied pixel (x_occ != x), mark all pixels till x_occ as free
                d = dy_abs;
                yi = y;
                for (int xi = x; xi != x_occ; xi += adder_x)
                {
                    map_current.at<char>(yi, xi) = 0;
                    if (d > 0) 
                    {
                        yi += adder_y;
                        d -= dx_abs;
                    }
                    d += dy_abs;
                }
            }
            else // Angle of ray is closer to y axis
            {
                int d = dx_abs;
                int xi = x;
                int y_occ = y;
                for (int yi = y; yi != (y+dy); yi += adder_y)
                {
                    // If the pixel is outside of the map then move on to the next ray
                    if (xi >= map_width || yi >= map_height || xi < 0 || yi < 0)
                        break;
                    // If found an occupied pixel, save endpoint, mark on current map and terminate ray
                    if (map_full.at<char>(yi, xi) == 100)
                    {
                        y_occ = yi;
                        map_current.at<char>(yi, xi) = 100;
                        break;
                    }
                    if (d > 0)
                    {
                        xi += adder_x;
                        d -= dy_abs;
                    }
                    d += dx_abs;
                }
                // If ray intersected an occupied pixel (y_occ != y), mark all pixels till y_occ as free
                d = dx_abs;
                xi = x;
                for (int yi = y; yi != y_occ; yi += adder_y)
                {
                    map_current.at<char>(yi, xi) = 0;
                    if (d > 0)
                    {
                        xi += adder_x;
                        d -= dy_abs;
                    }
                    d += dx_abs;
                }
            }
        }
        // Update costmap
        costmap_full.copyTo(costmap_current);
        cv::Mat mask = (map_current < 0);
        costmap_current.setTo(cv::Scalar::all(-1), mask);
        return true;
    }
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotSimServer>());
  rclcpp::shutdown();
  return 0;
}