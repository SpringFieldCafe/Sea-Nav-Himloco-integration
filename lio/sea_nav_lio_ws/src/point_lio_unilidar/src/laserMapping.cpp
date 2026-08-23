#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>


#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
// #include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "IMU_Processing.hpp"
#include "parameters.h"
#include "Estimator.h"

#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;

int feats_down_size = 0;
int time_log_counter = 0;
int scan_count = 0;
int publish_count = 0;

int frame_ct = 0;
double time_update_last = 0.0;
double time_current = 0.0;
double time_predict_last_const = 0.0;
double t_last = 0.0;

shared_ptr<ImuProcess> p_imu(new ImuProcess());
bool init_map = false;
bool flg_first_scan = true;
PointCloudXYZI::Ptr ptr_con(new PointCloudXYZI());

double T1[MAXN];
double s_plot[MAXN];
double s_plot2[MAXN];
double s_plot3[MAXN];
double s_plot11[MAXN];

double match_time = 0;
double solve_time = 0;
double propag_time = 0;
double update_time = 0;

bool lidar_pushed = false;
bool flg_reset = false;
bool flg_exit = false;

vector<BoxPointType> cub_needrm;

deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double> time_buffer;
deque<std::uint64_t> lidar_receive_order_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_deque;

std::uint64_t lidar_receive_order = 0;
std::uint64_t lidar_drop_count = 0;
std::uint64_t imu_drop_count = 0;

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

V3D euler_cur;

MeasureGroup Measures;

sensor_msgs::msg::Imu imu_last, imu_next;
sensor_msgs::msg::Imu::ConstSharedPtr imu_last_ptr;
nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;

std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br;

double lio_diag_last_report_time = 0.0;
bool lio_diag_scan_bbox_valid = false;
Eigen::Vector3d lio_diag_scan_bbox_min;
Eigen::Vector3d lio_diag_scan_bbox_max;
bool lio_diag_state_init_valid = false;
Eigen::Vector3d lio_diag_state_pos_init;

void record_startup_cloud_diagnostic(double timestamp, const PointCloudXYZI &cloud)
{
    if (!lio_diag.startup_input_started)
    {
        lio_diag.startup_input_started = true;
        lio_diag.startup_first_cloud_time = timestamp;
        lio_diag.startup_cloud_point_min = std::numeric_limits<std::uint64_t>::max();
    }
    const double elapsed = timestamp - lio_diag.startup_first_cloud_time;
    if (elapsed < 0.0 || elapsed > 10.0)
    {
        if (elapsed > 10.0)
            lio_diag.startup_input_window_complete = true;
        return;
    }

    lio_diag.startup_last_cloud_time = timestamp;
    lio_diag.startup_cloud_timestamps.push_back(timestamp);
    lio_diag.startup_cloud_count++;
    const std::uint64_t point_count = cloud.points.size();
    lio_diag.startup_cloud_point_sum += point_count;
    lio_diag.startup_cloud_point_min = std::min(lio_diag.startup_cloud_point_min, point_count);
    lio_diag.startup_cloud_point_max = std::max(lio_diag.startup_cloud_point_max, point_count);
    for (const auto &point : cloud.points)
    {
        const Eigen::Vector3d xyz(point.x, point.y, point.z);
        if (!xyz.allFinite())
            continue;
        lio_diag.startup_cloud_xyz_min = lio_diag.startup_cloud_xyz_min.cwiseMin(xyz);
        lio_diag.startup_cloud_xyz_max = lio_diag.startup_cloud_xyz_max.cwiseMax(xyz);
        lio_diag.startup_cloud_xyz_sum += xyz;
        lio_diag.startup_cloud_xyz_count++;
    }
}

void record_startup_imu_diagnostic(double timestamp, const sensor_msgs::msg::Imu &imu)
{
    if (!lio_diag.startup_input_started)
        return;
    const double elapsed = timestamp - lio_diag.startup_first_cloud_time;
    if (elapsed < 0.0 || elapsed > 10.0)
        return;
    const Eigen::Vector3d acc(
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);
    const Eigen::Vector3d gyro(
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
    if (!acc.allFinite() || !gyro.allFinite())
        return;
    lio_diag.startup_imu_count++;
    lio_diag.startup_imu_timestamps.push_back(timestamp);
    lio_diag.startup_imu_acc_sum += acc;
    lio_diag.startup_imu_acc_sq_sum += acc.cwiseProduct(acc);
    lio_diag.startup_imu_gyro_sum += gyro;
    lio_diag.startup_imu_gyro_sq_sum += gyro.cwiseProduct(gyro);
}

struct LioDiagStateSnapshot
{
    V3D pos = V3D::Zero();
    V3D vel = V3D::Zero();
    V3D euler_deg = V3D::Zero();
    V3D gravity = V3D::Zero();
    V3D acc_bias = V3D::Zero();
    V3D gyro_bias = V3D::Zero();
};

template<typename State>
LioDiagStateSnapshot snapshot_lio_state(const State &state)
{
    LioDiagStateSnapshot snapshot;
    snapshot.pos = state.pos;
    snapshot.vel = state.vel;
    snapshot.euler_deg = SO3ToEuler(state.rot);
    snapshot.gravity = state.gravity;
    snapshot.acc_bias = state.ba;
    snapshot.gyro_bias = state.bg;
    return snapshot;
}

enum LioBiasDiagnosticSource
{
    LIO_BIAS_SOURCE_PROPAGATION,
    LIO_BIAS_SOURCE_IMU_UPDATE,
    LIO_BIAS_SOURCE_LIDAR_UPDATE,
    LIO_BIAS_SOURCE_OTHER,
};

void begin_bias_diagnostic_frame()
{
    lio_diag.bias_diag_frame_active =
        lio_diag.bias_diag_init_valid && lio_diag.bias_diag_frame_count < 100;
    if (lio_diag.bias_diag_frame_active)
    {
        if (use_imu_as_input)
        {
            lio_diag.bias_diag_frame_start_gyro = kf_input.x_.bg;
            lio_diag.bias_diag_frame_start_acc = kf_input.x_.ba;
        }
        else
        {
            lio_diag.bias_diag_frame_start_gyro = kf_output.x_.bg;
            lio_diag.bias_diag_frame_start_acc = kf_output.x_.ba;
        }
        lio_diag.bias_diag_frame_prop_gyro.setZero();
        lio_diag.bias_diag_frame_prop_acc.setZero();
        lio_diag.bias_diag_frame_imu_gyro.setZero();
        lio_diag.bias_diag_frame_imu_acc.setZero();
        lio_diag.bias_diag_frame_lidar_gyro.setZero();
        lio_diag.bias_diag_frame_lidar_acc.setZero();
        lio_diag.bias_diag_frame_other_gyro.setZero();
        lio_diag.bias_diag_frame_other_acc.setZero();
    }
    if (lio_diag.bias_diag_post100_active)
    {
        if (use_imu_as_input)
        {
            lio_diag.bias_diag_post100_frame_start_gyro = kf_input.x_.bg;
            lio_diag.bias_diag_post100_frame_start_acc = kf_input.x_.ba;
        }
        else
        {
            lio_diag.bias_diag_post100_frame_start_gyro = kf_output.x_.bg;
            lio_diag.bias_diag_post100_frame_start_acc = kf_output.x_.ba;
        }
        lio_diag.bias_diag_post100_frame_imu_gyro.setZero();
        lio_diag.bias_diag_post100_frame_imu_acc.setZero();
        lio_diag.bias_diag_post100_frame_lidar_gyro.setZero();
        lio_diag.bias_diag_post100_frame_lidar_acc.setZero();
        lio_diag.bias_diag_post100_frame_prop_gyro.setZero();
        lio_diag.bias_diag_post100_frame_prop_acc.setZero();
        lio_diag.bias_diag_post100_frame_other_gyro.setZero();
        lio_diag.bias_diag_post100_frame_other_acc.setZero();
    }
}

double bias_diag_quantile(const std::vector<double> &values, double fraction)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> ordered = values;
    std::sort(ordered.begin(), ordered.end());
    const double position = fraction * static_cast<double>(ordered.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, ordered.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
}

void capture_bias_drift_snapshot_if_needed(const V3D &position,
                                           const V3D &velocity,
                                           const V3D &gyro_bias,
                                           const V3D &acc_bias)
{
    if (!lio_diag.bias_diag_post100_active)
        return;
    const double drift = (position - lio_diag.bias_diag_map_init_pos).norm();
    const double thresholds[3] = {0.2, 0.5, 1.0};
    for (int index = 0; index < 3; ++index)
    {
        auto &snapshot = lio_diag.bias_diag_drift_snapshots[index];
        if (snapshot.valid || drift < thresholds[index])
            continue;
        snapshot.valid = true;
        snapshot.frame = lio_diag.bias_diag_post100_frame_count;
        snapshot.position_drift = drift;
        snapshot.velocity_norm = velocity.norm();
        snapshot.acc_bias = acc_bias;
        snapshot.gyro_bias = gyro_bias;
        snapshot.acc_delta_from_frame100 = acc_bias - lio_diag.bias_diag_post100_baseline_acc;
        snapshot.acc_imu_delta_from_frame100 = lio_diag.bias_diag_post100_imu_acc;
        snapshot.acc_lidar_delta_from_frame100 = lio_diag.bias_diag_post100_lidar_acc;
        snapshot.gyro_delta_from_frame100 = gyro_bias - lio_diag.bias_diag_post100_baseline_gyro;
        snapshot.gyro_imu_delta_from_frame100 = lio_diag.bias_diag_post100_imu_gyro;
        snapshot.gyro_lidar_delta_from_frame100 = lio_diag.bias_diag_post100_lidar_gyro;
        snapshot.acc_unaccounted = lio_diag.bias_diag_post100_unaccounted_acc;
        snapshot.gyro_unaccounted = lio_diag.bias_diag_post100_unaccounted_gyro;
        snapshot.imu_acc_innovation_p50 = bias_diag_quantile(lio_diag.bias_diag_post100_imu_acc_innovations, 0.50);
        snapshot.imu_acc_innovation_p95 = bias_diag_quantile(lio_diag.bias_diag_post100_imu_acc_innovations, 0.95);
        snapshot.imu_gyro_innovation_p50 = bias_diag_quantile(lio_diag.bias_diag_post100_imu_gyro_innovations, 0.50);
        snapshot.imu_gyro_innovation_p95 = bias_diag_quantile(lio_diag.bias_diag_post100_imu_gyro_innovations, 0.95);
        snapshot.max_imu_acc_bias_delta_per_update = lio_diag.bias_diag_post100_max_imu_acc_delta;
        snapshot.max_imu_gyro_bias_delta_per_update = lio_diag.bias_diag_post100_max_imu_gyro_delta;
    }
    if (lio_diag.bias_diag_drift_snapshots[2].valid)
        lio_diag.bias_diag_post100_active = false;
}

void finish_bias_diagnostic_frame()
{
    if (!lio_diag.bias_diag_frame_active && !lio_diag.bias_diag_post100_active)
        return;

    V3D frame_end_gyro;
    V3D frame_end_acc;
    if (use_imu_as_input)
    {
        frame_end_gyro = kf_input.x_.bg;
        frame_end_acc = kf_input.x_.ba;
    }
    else
    {
        frame_end_gyro = kf_output.x_.bg;
        frame_end_acc = kf_output.x_.ba;
    }
    if (lio_diag.bias_diag_frame_active)
    {
        const V3D frame_total_gyro = frame_end_gyro - lio_diag.bias_diag_frame_start_gyro;
        const V3D frame_total_acc = frame_end_acc - lio_diag.bias_diag_frame_start_acc;
        const V3D frame_accounted_gyro =
            lio_diag.bias_diag_frame_prop_gyro + lio_diag.bias_diag_frame_imu_gyro +
            lio_diag.bias_diag_frame_lidar_gyro + lio_diag.bias_diag_frame_other_gyro;
        const V3D frame_accounted_acc =
            lio_diag.bias_diag_frame_prop_acc + lio_diag.bias_diag_frame_imu_acc +
            lio_diag.bias_diag_frame_lidar_acc + lio_diag.bias_diag_frame_other_acc;

        lio_diag.bias_diag_total_gyro += frame_total_gyro;
        lio_diag.bias_diag_total_acc += frame_total_acc;
        lio_diag.bias_diag_unaccounted_gyro += frame_total_gyro - frame_accounted_gyro;
        lio_diag.bias_diag_unaccounted_acc += frame_total_acc - frame_accounted_acc;
        lio_diag.bias_diag_frame_active = false;
        lio_diag.bias_diag_frame_count++;
        if (lio_diag.bias_diag_frame_count == 100)
        {
            lio_diag.bias_diag_end_gyro = frame_end_gyro;
            lio_diag.bias_diag_end_acc = frame_end_acc;
            lio_diag.bias_diag_post100_active = true;
            lio_diag.bias_diag_post100_frame_count = 0;
            lio_diag.bias_diag_post100_baseline_gyro = frame_end_gyro;
            lio_diag.bias_diag_post100_baseline_acc = frame_end_acc;
        }
    }

    if (lio_diag.bias_diag_post100_active)
    {
        const V3D frame_total_gyro = frame_end_gyro - lio_diag.bias_diag_post100_frame_start_gyro;
        const V3D frame_total_acc = frame_end_acc - lio_diag.bias_diag_post100_frame_start_acc;
        const V3D frame_accounted_gyro =
            lio_diag.bias_diag_post100_frame_prop_gyro + lio_diag.bias_diag_post100_frame_imu_gyro +
            lio_diag.bias_diag_post100_frame_lidar_gyro + lio_diag.bias_diag_post100_frame_other_gyro;
        const V3D frame_accounted_acc =
            lio_diag.bias_diag_post100_frame_prop_acc + lio_diag.bias_diag_post100_frame_imu_acc +
            lio_diag.bias_diag_post100_frame_lidar_acc + lio_diag.bias_diag_post100_frame_other_acc;
        lio_diag.bias_diag_post100_prop_gyro += lio_diag.bias_diag_post100_frame_prop_gyro;
        lio_diag.bias_diag_post100_prop_acc += lio_diag.bias_diag_post100_frame_prop_acc;
        lio_diag.bias_diag_post100_imu_gyro += lio_diag.bias_diag_post100_frame_imu_gyro;
        lio_diag.bias_diag_post100_imu_acc += lio_diag.bias_diag_post100_frame_imu_acc;
        lio_diag.bias_diag_post100_lidar_gyro += lio_diag.bias_diag_post100_frame_lidar_gyro;
        lio_diag.bias_diag_post100_lidar_acc += lio_diag.bias_diag_post100_frame_lidar_acc;
        lio_diag.bias_diag_post100_other_gyro += lio_diag.bias_diag_post100_frame_other_gyro;
        lio_diag.bias_diag_post100_other_acc += lio_diag.bias_diag_post100_frame_other_acc;
        lio_diag.bias_diag_post100_unaccounted_gyro += frame_total_gyro - frame_accounted_gyro;
        lio_diag.bias_diag_post100_unaccounted_acc += frame_total_acc - frame_accounted_acc;
        lio_diag.bias_diag_post100_frame_count++;

        V3D position;
        V3D velocity;
        if (use_imu_as_input)
        {
            position = kf_input.x_.pos;
            velocity = kf_input.x_.vel;
        }
        else
        {
            position = kf_output.x_.pos;
            velocity = kf_output.x_.vel;
        }
        capture_bias_drift_snapshot_if_needed(position, velocity, frame_end_gyro, frame_end_acc);
    }
}

void record_bias_diagnostic_delta(const LioDiagStateSnapshot &before,
                                 const LioDiagStateSnapshot &after,
                                 LioBiasDiagnosticSource source)
{
    if (!lio_diag.bias_diag_frame_active)
        return;

    const V3D gyro_delta = after.gyro_bias - before.gyro_bias;
    const V3D acc_delta = after.acc_bias - before.acc_bias;
    switch (source)
    {
    case LIO_BIAS_SOURCE_PROPAGATION:
        lio_diag.bias_diag_from_prop_gyro += gyro_delta;
        lio_diag.bias_diag_from_prop_acc += acc_delta;
        lio_diag.bias_diag_frame_prop_gyro += gyro_delta;
        lio_diag.bias_diag_frame_prop_acc += acc_delta;
        if (lio_diag.bias_diag_post100_active)
        {
            lio_diag.bias_diag_post100_frame_prop_gyro += gyro_delta;
            lio_diag.bias_diag_post100_frame_prop_acc += acc_delta;
        }
        break;
    case LIO_BIAS_SOURCE_IMU_UPDATE:
        lio_diag.bias_diag_from_imu_gyro += gyro_delta;
        lio_diag.bias_diag_from_imu_acc += acc_delta;
        lio_diag.bias_diag_frame_imu_gyro += gyro_delta;
        lio_diag.bias_diag_frame_imu_acc += acc_delta;
        if (lio_diag.bias_diag_post100_active)
        {
            lio_diag.bias_diag_post100_frame_imu_gyro += gyro_delta;
            lio_diag.bias_diag_post100_frame_imu_acc += acc_delta;
            lio_diag.bias_diag_post100_max_imu_gyro_delta =
                std::max(lio_diag.bias_diag_post100_max_imu_gyro_delta, gyro_delta.norm());
            lio_diag.bias_diag_post100_max_imu_acc_delta =
                std::max(lio_diag.bias_diag_post100_max_imu_acc_delta, acc_delta.norm());
        }
        lio_diag.bias_diag_max_imu_gyro_norm =
            std::max(lio_diag.bias_diag_max_imu_gyro_norm, gyro_delta.norm());
        lio_diag.bias_diag_max_imu_acc_norm =
            std::max(lio_diag.bias_diag_max_imu_acc_norm, acc_delta.norm());
        break;
    case LIO_BIAS_SOURCE_LIDAR_UPDATE:
        lio_diag.bias_diag_from_lidar_gyro += gyro_delta;
        lio_diag.bias_diag_from_lidar_acc += acc_delta;
        lio_diag.bias_diag_frame_lidar_gyro += gyro_delta;
        lio_diag.bias_diag_frame_lidar_acc += acc_delta;
        if (lio_diag.bias_diag_post100_active)
        {
            lio_diag.bias_diag_post100_frame_lidar_gyro += gyro_delta;
            lio_diag.bias_diag_post100_frame_lidar_acc += acc_delta;
        }
        lio_diag.bias_diag_max_lidar_gyro_norm =
            std::max(lio_diag.bias_diag_max_lidar_gyro_norm, gyro_delta.norm());
        lio_diag.bias_diag_max_lidar_acc_norm =
            std::max(lio_diag.bias_diag_max_lidar_acc_norm, acc_delta.norm());
        break;
    case LIO_BIAS_SOURCE_OTHER:
        lio_diag.bias_diag_from_other_gyro += gyro_delta;
        lio_diag.bias_diag_from_other_acc += acc_delta;
        lio_diag.bias_diag_frame_other_gyro += gyro_delta;
        lio_diag.bias_diag_frame_other_acc += acc_delta;
        if (lio_diag.bias_diag_post100_active)
        {
            lio_diag.bias_diag_post100_frame_other_gyro += gyro_delta;
            lio_diag.bias_diag_post100_frame_other_acc += acc_delta;
        }
        break;
    }
}

template<typename Filter>
void record_post100_imu_innovation(const Filter &filter)
{
    if (!lio_diag.bias_diag_post100_active || !filter.last_imu_update_valid())
        return;
    const auto &innovation = filter.last_imu_innovation();
    const double gyro_norm = innovation.template segment<3>(0).norm();
    const double acc_norm = innovation.template segment<3>(3).norm();
    if (std::isfinite(gyro_norm))
        lio_diag.bias_diag_post100_imu_gyro_innovations.push_back(gyro_norm);
    if (std::isfinite(acc_norm))
        lio_diag.bias_diag_post100_imu_acc_innovations.push_back(acc_norm);
}

void report_bias_at_drift_diagnostics()
{
    static const char *labels[3] = {"0P2", "0P5", "1P0"};
    for (int index = 0; index < 3; ++index)
    {
        auto &snapshot = lio_diag.bias_diag_drift_snapshots[index];
        if (!snapshot.valid || snapshot.reported)
            continue;
        std::cout << "[LIO-DIAG] BIAS_AT_DRIFT_" << labels[index]
                  << " FRAME=" << snapshot.frame
                  << " POSITION_DRIFT=" << snapshot.position_drift
                  << " VELOCITY_NORM=" << snapshot.velocity_norm
                  << " ACC_BIAS=" << snapshot.acc_bias.transpose()
                  << " GYRO_BIAS=" << snapshot.gyro_bias.transpose()
                  << " ACC_DELTA_FROM_FRAME100=" << snapshot.acc_delta_from_frame100.transpose()
                  << " ACC_IMU_DELTA_FROM_FRAME100=" << snapshot.acc_imu_delta_from_frame100.transpose()
                  << " ACC_LIDAR_DELTA_FROM_FRAME100=" << snapshot.acc_lidar_delta_from_frame100.transpose()
                  << " GYRO_DELTA_FROM_FRAME100=" << snapshot.gyro_delta_from_frame100.transpose()
                  << " GYRO_IMU_DELTA_FROM_FRAME100=" << snapshot.gyro_imu_delta_from_frame100.transpose()
                  << " GYRO_LIDAR_DELTA_FROM_FRAME100=" << snapshot.gyro_lidar_delta_from_frame100.transpose()
                  << " ACC_UNACCOUNTED=" << snapshot.acc_unaccounted.transpose()
                  << " GYRO_UNACCOUNTED=" << snapshot.gyro_unaccounted.transpose()
                  << " IMU_ACC_INNOVATION_NORM_P50=" << snapshot.imu_acc_innovation_p50
                  << " IMU_ACC_INNOVATION_NORM_P95=" << snapshot.imu_acc_innovation_p95
                  << " IMU_GYRO_INNOVATION_NORM_P50=" << snapshot.imu_gyro_innovation_p50
                  << " IMU_GYRO_INNOVATION_NORM_P95=" << snapshot.imu_gyro_innovation_p95
                  << " MAX_IMU_ACC_BIAS_DELTA_PER_UPDATE=" << snapshot.max_imu_acc_bias_delta_per_update
                  << " MAX_IMU_GYRO_BIAS_DELTA_PER_UPDATE=" << snapshot.max_imu_gyro_bias_delta_per_update
                  << std::endl;
        snapshot.reported = true;
    }
}

void report_startup_input_diagnostics()
{
    if (!lio_diag.startup_input_window_complete || lio_diag.startup_input_reported)
        return;

    const auto quantile = [](std::vector<double> values, double fraction) {
        if (values.empty())
            return std::numeric_limits<double>::quiet_NaN();
        std::sort(values.begin(), values.end());
        const double position = fraction * static_cast<double>(values.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(position);
        const std::size_t upper = std::min(lower + 1, values.size() - 1);
        const double weight = position - static_cast<double>(lower);
        return values[lower] * (1.0 - weight) + values[upper] * weight;
    };

    lio_diag.startup_cloud_imu_offsets.clear();
    for (const double cloud_time : lio_diag.startup_cloud_timestamps)
    {
        if (lio_diag.startup_imu_timestamps.empty())
            break;
        const auto upper = std::lower_bound(
            lio_diag.startup_imu_timestamps.begin(),
            lio_diag.startup_imu_timestamps.end(), cloud_time);
        double nearest = std::numeric_limits<double>::infinity();
        if (upper != lio_diag.startup_imu_timestamps.end())
            nearest = std::min(nearest, std::abs(*upper - cloud_time));
        if (upper != lio_diag.startup_imu_timestamps.begin())
        {
            const auto previous = std::prev(upper);
            nearest = std::min(nearest, std::abs(*previous - cloud_time));
        }
        if (std::isfinite(nearest))
            lio_diag.startup_cloud_imu_offsets.push_back(nearest);
    }

    const double cloud_point_mean = lio_diag.startup_cloud_count == 0
        ? std::numeric_limits<double>::quiet_NaN()
        : static_cast<double>(lio_diag.startup_cloud_point_sum) /
          static_cast<double>(lio_diag.startup_cloud_count);
    Eigen::Vector3d cloud_xyz_mean = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d imu_acc_mean = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d imu_gyro_mean = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d imu_acc_variance = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d imu_gyro_variance = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    if (lio_diag.startup_cloud_xyz_count > 0)
        cloud_xyz_mean = lio_diag.startup_cloud_xyz_sum /
                         static_cast<double>(lio_diag.startup_cloud_xyz_count);
    if (lio_diag.startup_imu_count > 0)
    {
        const double count = static_cast<double>(lio_diag.startup_imu_count);
        imu_acc_mean = lio_diag.startup_imu_acc_sum / count;
        imu_gyro_mean = lio_diag.startup_imu_gyro_sum / count;
        imu_acc_variance = (lio_diag.startup_imu_acc_sq_sum / count) -
                           imu_acc_mean.cwiseProduct(imu_acc_mean);
        imu_gyro_variance = (lio_diag.startup_imu_gyro_sq_sum / count) -
                            imu_gyro_mean.cwiseProduct(imu_gyro_mean);
    }

    const double init_duration = lio_diag.startup_map_initialized_time > 0.0
        ? lio_diag.startup_map_initialized_time - lio_diag.startup_first_cloud_time
        : std::numeric_limits<double>::quiet_NaN();
    const double cloud_imu_mean = lio_diag.startup_cloud_imu_offsets.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : std::accumulate(lio_diag.startup_cloud_imu_offsets.begin(),
                          lio_diag.startup_cloud_imu_offsets.end(), 0.0) /
          static_cast<double>(lio_diag.startup_cloud_imu_offsets.size());

    std::cout << "[LIO-DIAG] STARTUP_INPUT_10S"
              << " CLOUD_FIRST_TIME=" << lio_diag.startup_first_cloud_time
              << " CLOUD_LAST_TIME=" << lio_diag.startup_last_cloud_time
              << " CLOUD_TIMESTAMP_MONOTONIC=" << (std::is_sorted(lio_diag.startup_cloud_timestamps.begin(),
                                                                     lio_diag.startup_cloud_timestamps.end()) ? "PASS" : "FAIL")
              << " CLOUD_COUNT=" << lio_diag.startup_cloud_count
              << " CLOUD_POINT_COUNT_MIN=" << lio_diag.startup_cloud_point_min
              << " CLOUD_POINT_COUNT_MEAN=" << cloud_point_mean
              << " CLOUD_POINT_COUNT_MAX=" << lio_diag.startup_cloud_point_max
              << " CLOUD_XYZ_MIN=" << lio_diag.startup_cloud_xyz_min.transpose()
              << " CLOUD_XYZ_MAX=" << lio_diag.startup_cloud_xyz_max.transpose()
              << " CLOUD_XYZ_MEAN=" << cloud_xyz_mean.transpose()
              << " IMU_COUNT=" << lio_diag.startup_imu_count
              << " IMU_TIMESTAMP_MONOTONIC=" << (std::is_sorted(lio_diag.startup_imu_timestamps.begin(),
                                                                  lio_diag.startup_imu_timestamps.end()) ? "PASS" : "FAIL")
              << " IMU_ACC_MEAN=" << imu_acc_mean.transpose()
              << " IMU_ACC_STD=" << imu_acc_variance.cwiseMax(0.0).cwiseSqrt().transpose()
              << " IMU_GYRO_MEAN=" << imu_gyro_mean.transpose()
              << " IMU_GYRO_STD=" << imu_gyro_variance.cwiseMax(0.0).cwiseSqrt().transpose()
              << " MAP_INITIALIZED_TIME=" << lio_diag.startup_map_initialized_time
              << " MAP_INIT_DURATION_S=" << init_duration
              << " INITIAL_MAP_POINTS=" << lio_diag.startup_initial_map_points
              << " CLOUD_IMU_SYNC_SUCCESS=" << lio_diag.startup_sync_success_count
              << " CLOUD_IMU_OFFSET_COUNT=" << lio_diag.startup_cloud_imu_offsets.size()
              << " CLOUD_IMU_OFFSET_MEAN_S=" << cloud_imu_mean
              << " CLOUD_IMU_OFFSET_P95_S=" << quantile(lio_diag.startup_cloud_imu_offsets, 0.95)
              << " CLOUD_IMU_OFFSET_MAX_S=" << (lio_diag.startup_cloud_imu_offsets.empty()
                                                    ? std::numeric_limits<double>::quiet_NaN()
                                                    : *std::max_element(lio_diag.startup_cloud_imu_offsets.begin(),
                                                                        lio_diag.startup_cloud_imu_offsets.end()))
              << std::endl;
    lio_diag.startup_input_reported = true;
}

void record_bias_init_reset(const LioDiagStateSnapshot &before,
                            const LioDiagStateSnapshot &after)
{
    if (!lio_diag.bias_diag_origin_valid)
    {
        lio_diag.bias_diag_origin_gyro = before.gyro_bias;
        lio_diag.bias_diag_origin_acc = before.acc_bias;
        lio_diag.bias_diag_origin_valid = true;
    }
    lio_diag.bias_diag_init_reset_gyro += after.gyro_bias - before.gyro_bias;
    lio_diag.bias_diag_init_reset_acc += after.acc_bias - before.acc_bias;
}

template<typename State, typename Operation>
void run_bias_diagnostic_operation(State &state, Operation &&operation,
                                   LioBiasDiagnosticSource source)
{
    const LioDiagStateSnapshot before = snapshot_lio_state(state);
    operation();
    const LioDiagStateSnapshot after = snapshot_lio_state(state);
    record_bias_diagnostic_delta(before, after, source);
}

void report_bias_source_diagnostics()
{
    if (lio_diag.bias_diag_reported || !lio_diag.bias_diag_init_valid ||
        lio_diag.bias_diag_frame_count < 100)
        return;

    const V3D gyro_from_init = lio_diag.bias_diag_end_gyro - lio_diag.bias_diag_init_gyro;
    const V3D acc_from_init = lio_diag.bias_diag_end_acc - lio_diag.bias_diag_init_acc;
    std::cout << "[LIO-DIAG] BIAS_CONSERVATION_FIRST100"
              << " FRAMES=" << lio_diag.bias_diag_frame_count
              << " GYRO_TOTAL_DELTA="
              << (lio_diag.bias_diag_init_reset_gyro + lio_diag.bias_diag_total_gyro).transpose()
              << " GYRO_INIT_RESET_DELTA=" << lio_diag.bias_diag_init_reset_gyro.transpose()
              << " GYRO_BIAS_DELTA_FROM_INIT=" << gyro_from_init.transpose()
              << " GYRO_BIAS_DELTA_FROM_PROP=" << lio_diag.bias_diag_from_prop_gyro.transpose()
              << " GYRO_BIAS_DELTA_FROM_IMU_UPDATE=" << lio_diag.bias_diag_from_imu_gyro.transpose()
              << " GYRO_BIAS_DELTA_FROM_LIDAR_UPDATE=" << lio_diag.bias_diag_from_lidar_gyro.transpose()
              << " GYRO_BIAS_DELTA_FROM_OTHER=" << lio_diag.bias_diag_from_other_gyro.transpose()
              << " GYRO_UNACCOUNTED_DELTA=" << lio_diag.bias_diag_unaccounted_gyro.transpose()
              << " GYRO_TOTAL_FROM_END_STATE=" << gyro_from_init.transpose()
              << " ACC_TOTAL_DELTA="
              << (lio_diag.bias_diag_init_reset_acc + lio_diag.bias_diag_total_acc).transpose()
              << " ACC_INIT_RESET_DELTA=" << lio_diag.bias_diag_init_reset_acc.transpose()
              << " ACC_BIAS_DELTA_FROM_INIT=" << acc_from_init.transpose()
              << " ACC_BIAS_DELTA_FROM_PROP=" << lio_diag.bias_diag_from_prop_acc.transpose()
              << " ACC_BIAS_DELTA_FROM_IMU_UPDATE=" << lio_diag.bias_diag_from_imu_acc.transpose()
              << " ACC_BIAS_DELTA_FROM_LIDAR_UPDATE=" << lio_diag.bias_diag_from_lidar_acc.transpose()
              << " ACC_BIAS_DELTA_FROM_OTHER=" << lio_diag.bias_diag_from_other_acc.transpose()
              << " ACC_UNACCOUNTED_DELTA=" << lio_diag.bias_diag_unaccounted_acc.transpose()
              << " ACC_TOTAL_FROM_END_STATE=" << acc_from_init.transpose()
              << " MAX_IMU_UPDATE_GYRO_BIAS_DELTA=" << lio_diag.bias_diag_max_imu_gyro_norm
              << " MAX_LIDAR_UPDATE_GYRO_BIAS_DELTA=" << lio_diag.bias_diag_max_lidar_gyro_norm
              << " MAX_IMU_UPDATE_ACC_BIAS_DELTA=" << lio_diag.bias_diag_max_imu_acc_norm
              << " MAX_LIDAR_UPDATE_ACC_BIAS_DELTA=" << lio_diag.bias_diag_max_lidar_acc_norm
              << std::endl;
    lio_diag.bias_diag_reported = true;
}

bool estimate_local_scan_normal(const PointCloudXYZI &cloud, std::size_t index, V3D &normal)
{
    constexpr std::size_t kNeighbors = 6;
    if (cloud.size() < kNeighbors || index >= cloud.size())
        return false;

    const Eigen::Vector3d center_point(
        cloud.points[index].x, cloud.points[index].y, cloud.points[index].z);
    std::vector<std::pair<double, std::size_t>> distances;
    distances.reserve(cloud.size() - 1);
    for (std::size_t candidate = 0; candidate < cloud.size(); ++candidate)
    {
        if (candidate == index)
            continue;
        const Eigen::Vector3d point(
            cloud.points[candidate].x, cloud.points[candidate].y, cloud.points[candidate].z);
        distances.emplace_back((point - center_point).squaredNorm(), candidate);
    }
    if (distances.size() < kNeighbors - 1)
        return false;

    std::partial_sort(
        distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(kNeighbors - 1),
        distances.end());

    Eigen::Vector3d mean = center_point;
    for (std::size_t neighbor = 0; neighbor < kNeighbors - 1; ++neighbor)
    {
        const PointType &point = cloud.points[distances[neighbor].second];
        mean += Eigen::Vector3d(point.x, point.y, point.z);
    }
    mean /= static_cast<double>(kNeighbors);

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    covariance += (center_point - mean) * (center_point - mean).transpose();
    for (std::size_t neighbor = 0; neighbor < kNeighbors - 1; ++neighbor)
    {
        const PointType &point = cloud.points[distances[neighbor].second];
        const Eigen::Vector3d delta = Eigen::Vector3d(point.x, point.y, point.z) - mean;
        covariance += delta * delta.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success)
        return false;
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();
    if (!eigenvalues.allFinite() || eigenvalues(2) <= 1e-10)
        return false;

    // Reject nearly isotropic neighborhoods; they do not provide a useful
    // local plane direction for this diagnostic.
    if (eigenvalues(0) / eigenvalues(2) > 0.25)
        return false;

    normal = solver.eigenvectors().col(0);
    return normal.allFinite() && normal.norm() > 1e-9;
}

void record_prematch_scan_geometry()
{
    lio_diag_prematch_axes.assign(feats_down_size, -1);
    for (std::size_t index = 0; index < feats_down_body->size(); ++index)
    {
        V3D normal;
        if (!estimate_local_scan_normal(*feats_down_body, index, normal))
            continue;
        lio_diag_prematch_axes[index] = lio_diag_dominant_normal_axis(normal);
        lio_diag_record_prematch_normal(normal);
    }
}

void report_geometry_retention_diagnostics()
{
    const auto ratio = [](std::uint64_t value, std::uint64_t total) {
        return total == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(total);
    };
    const auto candidate_count = [&ratio](int axis) {
        return lio_diag.geometry_candidate_count[axis];
    };
    const auto outcome_count = [](int axis, int outcome) {
        return lio_diag.geometry_outcome_count[axis][outcome];
    };
    const auto accept_rate = [&ratio, &candidate_count, &outcome_count](int axis) {
        return ratio(outcome_count(axis, LIO_GEOMETRY_ACCEPTED), candidate_count(axis));
    };
    const auto print_axis = [&candidate_count, &outcome_count, &accept_rate](const char *name, int axis) {
        std::cout << " " << name << "_CANDIDATES=" << candidate_count(axis)
                  << " " << name << "_NEAREST_REJECT=" << outcome_count(axis, LIO_GEOMETRY_NEAREST_REJECT)
                  << " " << name << "_PLANE_REJECT=" << outcome_count(axis, LIO_GEOMETRY_PLANE_REJECT)
                  << " " << name << "_RESIDUAL_REJECT=" << outcome_count(axis, LIO_GEOMETRY_RESIDUAL_REJECT)
                  << " " << name << "_ACCEPTED=" << outcome_count(axis, LIO_GEOMETRY_ACCEPTED)
                  << " " << name << "_ACCEPT_RATE=" << accept_rate(axis);
    };

    std::cout << "[LIO-DIAG] GEOMETRY_RETENTION";
    print_axis("X", LIO_GEOMETRY_AXIS_X);
    print_axis("Y", LIO_GEOMETRY_AXIS_Y);
    print_axis("Z", LIO_GEOMETRY_AXIS_Z);
    std::cout << " UNLABELED_CANDIDATES=" << lio_diag.geometry_unlabeled_candidates
              << " UNLABELED_OUTCOMES=["
              << lio_diag.geometry_unlabeled_outcome_count[LIO_GEOMETRY_NEAREST_REJECT] << ","
              << lio_diag.geometry_unlabeled_outcome_count[LIO_GEOMETRY_PLANE_REJECT] << ","
              << lio_diag.geometry_unlabeled_outcome_count[LIO_GEOMETRY_RESIDUAL_REJECT] << ","
              << lio_diag.geometry_unlabeled_outcome_count[LIO_GEOMETRY_ACCEPTED] << ","
              << lio_diag.geometry_unlabeled_outcome_count[LIO_GEOMETRY_OTHER_REJECT] << "]"
              << std::endl;
}

void report_early_geometry_diagnostics()
{
    const auto ratio = [](std::uint64_t value, std::uint64_t total) {
        return total == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(total);
    };
    const auto quantile = [](const std::vector<double> &values, double fraction) {
        if (values.empty())
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> ordered = values;
        std::sort(ordered.begin(), ordered.end());
        const double position = fraction * static_cast<double>(ordered.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(position);
        const std::size_t upper = std::min(lower + 1, ordered.size() - 1);
        const double weight = position - static_cast<double>(lower);
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
    };
    const auto candidate_count = [](int axis) {
        return lio_diag.early_geometry_candidate_count[axis];
    };
    const auto outcome_count = [](int axis, int outcome) {
        return lio_diag.early_geometry_outcome_count[axis][outcome];
    };
    const auto accept_rate = [&ratio, &candidate_count, &outcome_count](int axis) {
        return ratio(outcome_count(axis, LIO_GEOMETRY_ACCEPTED), candidate_count(axis));
    };
    const auto print_axis = [&](const char *name, int axis) {
        std::cout << " " << name << "_CANDIDATES=" << candidate_count(axis)
                  << " " << name << "_NEAREST_REJECT=" << outcome_count(axis, LIO_GEOMETRY_NEAREST_REJECT)
                  << " " << name << "_PLANE_REJECT=" << outcome_count(axis, LIO_GEOMETRY_PLANE_REJECT)
                  << " " << name << "_RESIDUAL_REJECT=" << outcome_count(axis, LIO_GEOMETRY_RESIDUAL_REJECT)
                  << " " << name << "_ACCEPTED=" << outcome_count(axis, LIO_GEOMETRY_ACCEPTED)
                  << " " << name << "_ACCEPT_RATE=" << accept_rate(axis)
                  << " " << name << "_ABS_RESIDUAL_P50="
                  << quantile(lio_diag.early_abs_residuals[axis], 0.50)
                  << " " << name << "_ABS_RESIDUAL_P95="
                  << quantile(lio_diag.early_abs_residuals[axis], 0.95);
    };

    std::cout << "[LIO-DIAG] EARLY_GEOMETRY_FIRST200";
    print_axis("X", LIO_GEOMETRY_AXIS_X);
    print_axis("Y", LIO_GEOMETRY_AXIS_Y);
    print_axis("Z", LIO_GEOMETRY_AXIS_Z);
    std::cout << " UNLABELED_CANDIDATES=" << lio_diag.early_geometry_unlabeled_candidates
              << " UNLABELED_OUTCOMES=["
              << lio_diag.early_geometry_unlabeled_outcome_count[LIO_GEOMETRY_NEAREST_REJECT] << ","
              << lio_diag.early_geometry_unlabeled_outcome_count[LIO_GEOMETRY_PLANE_REJECT] << ","
              << lio_diag.early_geometry_unlabeled_outcome_count[LIO_GEOMETRY_RESIDUAL_REJECT] << ","
              << lio_diag.early_geometry_unlabeled_outcome_count[LIO_GEOMETRY_ACCEPTED] << ","
              << lio_diag.early_geometry_unlabeled_outcome_count[LIO_GEOMETRY_OTHER_REJECT] << "]"
              << std::endl;
}

std::uint64_t lio_diag_trace_count = 0;
V3D lio_diag_cum_prop_vel = V3D::Zero();
V3D lio_diag_cum_ekf_vel = V3D::Zero();
V3D lio_diag_cum_prop_rot = V3D::Zero();
V3D lio_diag_cum_ekf_rot = V3D::Zero();
V3D lio_diag_cum_ekf_acc_bias = V3D::Zero();
V3D lio_diag_cum_ekf_gyro_bias = V3D::Zero();
double lio_diag_cum_prop_vel_norm = 0.0;
double lio_diag_cum_ekf_vel_norm = 0.0;
double lio_diag_cum_prop_rot_norm = 0.0;
double lio_diag_cum_ekf_rot_norm = 0.0;

void record_ekf_trace(const LioDiagStateSnapshot &pre,
                      const LioDiagStateSnapshot &propagated,
                      const LioDiagStateSnapshot &post,
                      bool update_ok)
{
    if (lio_diag_trace_count >= 200)
        return;

    const V3D delta_prop_pos = propagated.pos - pre.pos;
    const V3D delta_prop_vel = propagated.vel - pre.vel;
    const V3D delta_prop_rot = propagated.euler_deg - pre.euler_deg;
    const V3D delta_ekf_pos = post.pos - propagated.pos;
    const V3D delta_ekf_vel = post.vel - propagated.vel;
    const V3D delta_ekf_rot = post.euler_deg - propagated.euler_deg;
    const V3D delta_ekf_acc_bias = post.acc_bias - propagated.acc_bias;
    const V3D delta_ekf_gyro_bias = post.gyro_bias - propagated.gyro_bias;

    lio_diag_cum_prop_vel += delta_prop_vel;
    lio_diag_cum_ekf_vel += delta_ekf_vel;
    lio_diag_cum_prop_rot += delta_prop_rot;
    lio_diag_cum_ekf_rot += delta_ekf_rot;
    lio_diag_cum_ekf_acc_bias += delta_ekf_acc_bias;
    lio_diag_cum_ekf_gyro_bias += delta_ekf_gyro_bias;
    lio_diag_cum_prop_vel_norm += delta_prop_vel.norm();
    lio_diag_cum_ekf_vel_norm += delta_ekf_vel.norm();
    lio_diag_cum_prop_rot_norm += delta_prop_rot.norm();
    lio_diag_cum_ekf_rot_norm += delta_ekf_rot.norm();

    std::cout << "[LIO-DIAG] EKF_TRACE"
              << " index=" << lio_diag_trace_count
              << " update=" << (update_ok ? "SUCCESS" : "FAIL")
              << " PRE_POS=" << pre.pos.transpose()
              << " PRE_VEL=" << pre.vel.transpose()
              << " PRE_ROT_DEG=" << pre.euler_deg.transpose()
              << " PRE_ACC_BIAS=" << pre.acc_bias.transpose()
              << " PRE_GYRO_BIAS=" << pre.gyro_bias.transpose()
              << " PROP_POS=" << propagated.pos.transpose()
              << " PROP_VEL=" << propagated.vel.transpose()
              << " PROP_ROT_DEG=" << propagated.euler_deg.transpose()
              << " PROP_ACC_BIAS=" << propagated.acc_bias.transpose()
              << " PROP_GYRO_BIAS=" << propagated.gyro_bias.transpose()
              << " POST_POS=" << post.pos.transpose()
              << " POST_VEL=" << post.vel.transpose()
              << " POST_ROT_DEG=" << post.euler_deg.transpose()
              << " POST_ACC_BIAS=" << post.acc_bias.transpose()
              << " POST_GYRO_BIAS=" << post.gyro_bias.transpose()
              << " DELTA_PROP_POS=" << delta_prop_pos.transpose()
              << " DELTA_PROP_VEL=" << delta_prop_vel.transpose()
              << " DELTA_PROP_ROT_DEG=" << delta_prop_rot.transpose()
              << " DELTA_EKF_POS=" << delta_ekf_pos.transpose()
              << " DELTA_EKF_VEL=" << delta_ekf_vel.transpose()
              << " DELTA_EKF_ROT_DEG=" << delta_ekf_rot.transpose()
              << " DELTA_EKF_ACC_BIAS=" << delta_ekf_acc_bias.transpose()
              << " DELTA_EKF_GYRO_BIAS=" << delta_ekf_gyro_bias.transpose()
              << std::endl;
    lio_diag_trace_count++;
}

template<typename Filter>
void record_early_ekf_gain(const Filter &filter)
{
    if (lio_diag.early_ekf_gain_attempts >= 50)
        return;

    lio_diag.early_ekf_gain_attempts++;
    if (!filter.last_update_valid())
        return;

    lio_diag.early_ekf_gain_samples++;
    const auto &dx = filter.last_update_dx();
    lio_diag.early_ekf_rot_delta_deg.push_back(dx.template segment<3>(3).norm() * 180.0 / M_PI);
    lio_diag.early_ekf_pos_delta_m.push_back(dx.template segment<3>(0).norm());
    lio_diag.early_ekf_k_rot_norm.push_back(filter.last_update_k_rot_norm());
    lio_diag.early_ekf_effective_r.push_back(filter.last_update_measurement_noise());

    const auto &p_before = filter.last_update_p_before();
    const auto &p_after = filter.last_update_p_after();
    if (!lio_diag.early_ekf_cov_start_valid)
    {
        lio_diag.early_ekf_p_rot_diag_start = p_before.template block<3, 3>(3, 3).diagonal();
        lio_diag.early_ekf_p_pos_diag_start = p_before.template block<3, 3>(0, 0).diagonal();
        lio_diag.early_ekf_cov_start_valid = true;
    }
    lio_diag.early_ekf_p_rot_diag_end = p_after.template block<3, 3>(3, 3).diagonal();
    lio_diag.early_ekf_p_pos_diag_end = p_after.template block<3, 3>(0, 0).diagonal();

    for (const double residual : filter.last_update_residual_abs())
    {
        if (std::isfinite(residual))
            lio_diag.early_ekf_abs_residuals.push_back(residual);
    }
}

void report_early_ekf_gain_diagnostics()
{
    if (lio_diag.early_ekf_gain_reported || lio_diag.early_ekf_gain_samples == 0)
        return;
    if (lio_diag.early_ekf_gain_attempts < 50)
        return;

    const auto quantile = [](const std::vector<double> &values, double fraction) {
        if (values.empty())
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> ordered = values;
        std::sort(ordered.begin(), ordered.end());
        const double position = fraction * static_cast<double>(ordered.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(position);
        const std::size_t upper = std::min(lower + 1, ordered.size() - 1);
        const double weight = position - static_cast<double>(lower);
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
    };

    std::cout << "[LIO-DIAG] EARLY_EKF_GAIN_FIRST50"
              << " ATTEMPTS=" << lio_diag.early_ekf_gain_attempts
              << " SAMPLES=" << lio_diag.early_ekf_gain_samples
              << " RESIDUAL_ABS_P50=" << quantile(lio_diag.early_ekf_abs_residuals, 0.50)
              << " RESIDUAL_ABS_P95=" << quantile(lio_diag.early_ekf_abs_residuals, 0.95)
              << " ROT_DELTA_PER_UPDATE_P50_DEG=" << quantile(lio_diag.early_ekf_rot_delta_deg, 0.50)
              << " ROT_DELTA_PER_UPDATE_P95_DEG=" << quantile(lio_diag.early_ekf_rot_delta_deg, 0.95)
              << " ROT_DELTA_PER_UPDATE_MAX_DEG="
              << (lio_diag.early_ekf_rot_delta_deg.empty()
                      ? std::numeric_limits<double>::quiet_NaN()
                      : *std::max_element(lio_diag.early_ekf_rot_delta_deg.begin(), lio_diag.early_ekf_rot_delta_deg.end()))
              << " POS_DELTA_PER_UPDATE_P50_M=" << quantile(lio_diag.early_ekf_pos_delta_m, 0.50)
              << " POS_DELTA_PER_UPDATE_P95_M=" << quantile(lio_diag.early_ekf_pos_delta_m, 0.95)
              << " K_ROT_NORM_P50=" << quantile(lio_diag.early_ekf_k_rot_norm, 0.50)
              << " K_ROT_NORM_P95=" << quantile(lio_diag.early_ekf_k_rot_norm, 0.95)
              << " P_ROT_DIAG_START=[" << lio_diag.early_ekf_p_rot_diag_start.transpose() << "]"
              << " P_ROT_DIAG_END=[" << lio_diag.early_ekf_p_rot_diag_end.transpose() << "]"
              << " P_POS_DIAG_START=[" << lio_diag.early_ekf_p_pos_diag_start.transpose() << "]"
              << " P_POS_DIAG_END=[" << lio_diag.early_ekf_p_pos_diag_end.transpose() << "]"
              << " R_EFFECTIVE_P50=" << quantile(lio_diag.early_ekf_effective_r, 0.50)
              << " R_EFFECTIVE_P95=" << quantile(lio_diag.early_ekf_effective_r, 0.95)
              << std::endl;
    lio_diag.early_ekf_gain_reported = true;
}

void update_lio_diag_scan_bbox()
{
    if (feats_down_world == nullptr || feats_down_world->empty())
        return;

    lio_diag_scan_bbox_min << std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
    lio_diag_scan_bbox_max << -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity();
    for (const auto &point : feats_down_world->points)
    {
        lio_diag_scan_bbox_min.x() = std::min(lio_diag_scan_bbox_min.x(), static_cast<double>(point.x));
        lio_diag_scan_bbox_min.y() = std::min(lio_diag_scan_bbox_min.y(), static_cast<double>(point.y));
        lio_diag_scan_bbox_min.z() = std::min(lio_diag_scan_bbox_min.z(), static_cast<double>(point.z));
        lio_diag_scan_bbox_max.x() = std::max(lio_diag_scan_bbox_max.x(), static_cast<double>(point.x));
        lio_diag_scan_bbox_max.y() = std::max(lio_diag_scan_bbox_max.y(), static_cast<double>(point.y));
        lio_diag_scan_bbox_max.z() = std::max(lio_diag_scan_bbox_max.z(), static_cast<double>(point.z));
    }
    lio_diag_scan_bbox_valid = true;
}

void get_lio_diag_state(Eigen::Vector3d &pos, Eigen::Vector3d &vel,
                        Eigen::Vector3d &gravity, Eigen::Vector3d &acc_bias,
                        Eigen::Vector3d &gyro_bias, V3D &euler_deg)
{
    if (use_imu_as_input)
    {
        pos = kf_input.x_.pos;
        vel = kf_input.x_.vel;
        gravity = kf_input.x_.gravity;
        acc_bias = kf_input.x_.ba;
        gyro_bias = kf_input.x_.bg;
        euler_deg = SO3ToEuler(kf_input.x_.rot);
    }
    else
    {
        pos = kf_output.x_.pos;
        vel = kf_output.x_.vel;
        gravity = kf_output.x_.gravity;
        acc_bias = kf_output.x_.ba;
        gyro_bias = kf_output.x_.bg;
        euler_deg = SO3ToEuler(kf_output.x_.rot);
    }
}

void report_lidar_information_update(const LioDiagnosticCounters::RollingMeasurementGroup &group,
                                     bool update_ok)
{
    const auto &hth_eigenvalues = use_imu_as_input
        ? kf_input.last_update_hth_eigenvalues()
        : kf_output.last_update_hth_eigenvalues();
    const double hth_condition = use_imu_as_input
        ? kf_input.last_update_hth_condition()
        : kf_output.last_update_hth_condition();
    const double hth_min_eigenvalue = use_imu_as_input
        ? kf_input.last_update_hth_min_eigenvalue()
        : kf_output.last_update_hth_min_eigenvalue();
    const int hth_rank = use_imu_as_input
        ? kf_input.last_update_hth_rank()
        : kf_output.last_update_hth_rank();
    std::cout << "[LIO-DIAG] INFORMATION_UPDATE"
              << " LIDAR_FRAME_ID=" << lio_diag.lidar_frame_id
              << " UPDATE_INDEX=" << lio_diag.frame_diag_ekf_call_count
              << " EKF_SUCCESS=" << (update_ok ? "YES" : "NO")
              << " MEASUREMENT_COUNT=" << group.accepted_point_count
              << " HTH_EIGENVALUES=";
    for (Eigen::Index i = 0; i < hth_eigenvalues.size(); ++i)
    {
        if (i > 0)
            std::cout << ",";
        std::cout << hth_eigenvalues(i);
    }
    const double accepted_count = static_cast<double>(group.accepted_point_count);
    const double signed_residual_mean = accepted_count > 0.0
        ? lio_diag.current_signed_residual_sum / accepted_count
        : std::numeric_limits<double>::quiet_NaN();
    std::cout << " HTH_CONDITION_NUMBER=" << hth_condition
              << " HTH_MIN_EIGENVALUE=" << hth_min_eigenvalue
              << " HTH_RANK=" << hth_rank
              << " HTH_NEAR_ZERO_COUNT=" << (hth_eigenvalues.size() - hth_rank)
              << " SIGNED_RESIDUAL_MEAN=" << signed_residual_mean
              << " SIGNED_RESIDUAL_NORMAL_PROJECTION_MEAN=" << signed_residual_mean
              << " ABS_RESIDUAL_MEAN=" << group.accepted_abs_residual_mean
              << " ABS_RESIDUAL_P95=" << group.accepted_abs_residual_p95
              << " DELTA_POSITION_NORM=" << group.delta_position_norm
              << " DELTA_VELOCITY_NORM=" << group.delta_velocity_norm
              << " DELTA_ROTATION_DEG=" << group.delta_rotation_deg
              << std::endl;
}

void report_startup_group_summary()
{
    if (lio_diag.startup_group_summary_reported || lio_diag.startup_group_count == 0)
        return;
    const auto first_success = lio_diag.startup_first_success_group == std::numeric_limits<std::uint64_t>::max()
        ? -1LL
        : static_cast<long long>(lio_diag.startup_first_success_group);
    std::cout << "[LIO-DIAG] STARTUP_GROUP_SUMMARY"
              << " FIRST_SUCCESS_GROUP=" << first_success
              << " GROUPS_WITH_ZERO_IMU=" << lio_diag.startup_groups_with_zero_imu
              << " GROUPS_WITH_LARGE_IMU_CLOUD_GAP=" << lio_diag.startup_groups_with_large_imu_cloud_gap
              << " MAX_IMU_CLOUD_GAP_S=" << lio_diag.startup_max_imu_cloud_gap_s
              << " LARGE_GAP_THRESHOLD_S=0.01"
              << " CLOUD_DROP_COUNT=" << lidar_drop_count
              << " IMU_DROP_COUNT=" << imu_drop_count
              << std::endl;
    lio_diag.startup_group_summary_reported = true;
}

double rolling_quantile(const std::vector<double> &values, double fraction)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> ordered = values;
    std::sort(ordered.begin(), ordered.end());
    const double position = fraction * static_cast<double>(ordered.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, ordered.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
}

void print_rolling_group(const char *prefix, const LioDiagnosticCounters::RollingMeasurementGroup &group)
{
    std::cout << prefix
              << " GROUP_ID=" << group.group_id
              << " LIDAR_BEGIN_TIME=" << group.lidar_begin_time
              << " LIDAR_END_TIME=" << group.lidar_end_time
              << " CLOUD_POINT_COUNT=" << group.cloud_point_count
              << " EKF_ATTEMPTS=" << group.ekf_attempts
              << " EKF_SUCCESS=" << group.ekf_success
              << " EFFECT_NUM=" << group.effect_num
              << " NEAREST_TOO_FEW=" << group.nearest_too_few
              << " PLANE_REJECT=" << group.plane_reject
              << " RESIDUAL_REJECT=" << group.residual_reject
              << " ACCEPTED_POINT_COUNT=" << group.accepted_point_count
              << " ACCEPTED_RESIDUAL_SIGNED_MEAN=" << group.accepted_signed_residual_mean
              << " ACCEPTED_RESIDUAL_ABS_MEAN=" << group.accepted_abs_residual_mean
              << " ACCEPTED_RESIDUAL_ABS_P50=" << group.accepted_abs_residual_p50
              << " ACCEPTED_RESIDUAL_ABS_P95=" << group.accepted_abs_residual_p95
              << " ACCEPTED_RESIDUAL_ABS_MAX=" << group.accepted_abs_residual_max
              << " NORMAL_MEAN_XYZ=" << group.normal_mean.transpose()
              << " NORMAL_ABS_MEAN_XYZ=" << group.normal_abs_mean.transpose()
              << " NORMAL_X_DOMINANT_COUNT=" << group.normal_x_dominant_count
              << " NORMAL_Y_DOMINANT_COUNT=" << group.normal_y_dominant_count
              << " NORMAL_Z_DOMINANT_COUNT=" << group.normal_z_dominant_count
              << " DELTA_POSITION_NORM=" << group.delta_position_norm
              << " DELTA_ROTATION_DEG=" << group.delta_rotation_deg
              << " DELTA_VELOCITY_NORM=" << group.delta_velocity_norm
              << " DELTA_GYRO_BIAS_NORM=" << group.delta_gyro_bias_norm
              << " DELTA_ACC_BIAS_NORM=" << group.delta_acc_bias_norm
              << " PRE_POSITION=" << group.pre_position.transpose()
              << " POST_POSITION=" << group.post_position.transpose()
              << " PRE_VELOCITY=" << group.pre_velocity.transpose()
              << " POST_VELOCITY=" << group.post_velocity.transpose()
              << std::endl;
}

void report_rolling_context(const char *header)
{
    std::cout << header << " COUNT=" << lio_diag.rolling_groups.size() << std::endl;
    for (const auto &group : lio_diag.rolling_groups)
        print_rolling_group("[LIO-DIAG] ROLLING_CONTEXT_GROUP", group);
}

void report_stable_exit_context()
{
    if (lio_diag.stable_exit_context_reported || lio_diag.drift_0p2_reported ||
        lio_diag.rolling_groups.empty())
        return;
    lio_diag.stable_exit_context_reported = true;
    report_rolling_context("[LIO-DIAG] STABLE_EXIT_CONTEXT");
}

void begin_lidar_frame_diagnostic(const MeasureGroup &meas)
{
    lio_diag.lidar_frame_id++;
    lio_diag.frame_diag_active = true;
    lio_diag.frame_diag_lidar_begin_time = meas.lidar_beg_time;
    lio_diag.frame_diag_lidar_end_time = meas.lidar_last_time;
    lio_diag.frame_diag_point_count = meas.lidar ? meas.lidar->points.size() : 0;
    lio_diag.frame_diag_ekf_call_count = 0;
    lio_diag.frame_diag_ekf_success_any = false;
    lio_diag.frame_diag_total_effect_num = 0;
    lio_diag.frame_diag_total_accepted_points = 0;
    lio_diag.frame_diag_total_nearest_reject = 0;
    lio_diag.frame_diag_total_plane_reject = 0;
    lio_diag.frame_diag_total_residual_reject = 0;
    lio_diag.frame_diag_hth_sum.setZero();
    lio_diag.frame_diag_signed_residual_sum = 0.0;
    lio_diag.frame_diag_signed_residual_sq_sum = 0.0;
    lio_diag.frame_diag_abs_residuals.clear();
    lio_diag.frame_diag_state_valid = false;
    lio_diag.frame_diag_delta_rotation_deg = 0.0;
}

void report_lidar_frame_diagnostic()
{
    if (!lio_diag.frame_diag_active)
        return;
    const V3D delta_position = lio_diag.frame_diag_post_position - lio_diag.frame_diag_pre_position;
    const V3D delta_velocity = lio_diag.frame_diag_post_velocity - lio_diag.frame_diag_pre_velocity;
    std::cout << "[LIO-DIAG] FRAME_DIAG"
              << " LIDAR_FRAME_ID=" << lio_diag.lidar_frame_id
              << " LIDAR_BEGIN_TIME=" << lio_diag.frame_diag_lidar_begin_time
              << " LIDAR_END_TIME=" << lio_diag.frame_diag_lidar_end_time
              << " POINT_COUNT=" << lio_diag.frame_diag_point_count
              << " EKF_CALL_COUNT=" << lio_diag.frame_diag_ekf_call_count
              << " EKF_SUCCESS_ANY=" << (lio_diag.frame_diag_ekf_success_any ? "YES" : "NO")
              << " TOTAL_EFFECT_NUM=" << lio_diag.frame_diag_total_effect_num
              << " TOTAL_ACCEPTED_POINTS=" << lio_diag.frame_diag_total_accepted_points
              << " TOTAL_NEAREST_REJECT=" << lio_diag.frame_diag_total_nearest_reject
              << " TOTAL_PLANE_REJECT=" << lio_diag.frame_diag_total_plane_reject
              << " TOTAL_RESIDUAL_REJECT=" << lio_diag.frame_diag_total_residual_reject
              << " PRE_POSITION=" << lio_diag.frame_diag_pre_position.transpose()
              << " POST_POSITION=" << lio_diag.frame_diag_post_position.transpose()
              << " PRE_VELOCITY=" << lio_diag.frame_diag_pre_velocity.transpose()
              << " POST_VELOCITY=" << lio_diag.frame_diag_post_velocity.transpose()
              << " FRAME_DELTA_POSITION_NORM=" << delta_position.norm()
              << " FRAME_DELTA_ROTATION_DEG=" << lio_diag.frame_diag_delta_rotation_deg
              << " FRAME_DELTA_VELOCITY_NORM=" << delta_velocity.norm()
              << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> frame_hth_solver(
        lio_diag.frame_diag_hth_sum);
    const Eigen::Matrix<double, 12, 1> frame_eigenvalues = frame_hth_solver.eigenvalues();
    const double frame_max_eigenvalue = frame_eigenvalues.maxCoeff();
    const double frame_rank_threshold = std::max(
        std::abs(frame_max_eigenvalue) * 1e-12, 1e-15);
    int frame_rank = 0;
    double frame_min_positive = std::numeric_limits<double>::quiet_NaN();
    for (Eigen::Index i = 0; i < frame_eigenvalues.size(); ++i)
    {
        if (frame_eigenvalues(i) > frame_rank_threshold)
        {
            ++frame_rank;
            if (!std::isfinite(frame_min_positive))
                frame_min_positive = frame_eigenvalues(i);
        }
    }
    const double frame_condition = frame_rank > 0
        ? frame_max_eigenvalue / frame_min_positive
        : std::numeric_limits<double>::quiet_NaN();
    const double accepted_count = static_cast<double>(lio_diag.frame_diag_total_accepted_points);
    const double signed_mean = accepted_count > 0.0
        ? lio_diag.frame_diag_signed_residual_sum / accepted_count
        : std::numeric_limits<double>::quiet_NaN();
    const double signed_variance = accepted_count > 0.0
        ? std::max(0.0, lio_diag.frame_diag_signed_residual_sq_sum / accepted_count - signed_mean * signed_mean)
        : std::numeric_limits<double>::quiet_NaN();
    std::cout << "[LIO-DIAG] FRAME_INFORMATION"
              << " FRAME_ID=" << lio_diag.lidar_frame_id
              << " ACCEPTED_COUNT=" << lio_diag.frame_diag_total_accepted_points
              << " EIGENVALUES=";
    for (Eigen::Index i = 0; i < frame_eigenvalues.size(); ++i)
    {
        if (i > 0)
            std::cout << ",";
        std::cout << frame_eigenvalues(i);
    }
    std::cout << " CONDITION_NUMBER=" << frame_condition
              << " MIN_EIGENVALUE=" << frame_min_positive
              << " RANK=" << frame_rank
              << " NEAR_ZERO_COUNT=" << (frame_eigenvalues.size() - frame_rank)
              << " SIGNED_MEAN=" << signed_mean
              << " SIGNED_STD=" << std::sqrt(signed_variance)
              << " ABS_MEAN=" << (accepted_count > 0.0
                  ? std::accumulate(lio_diag.frame_diag_abs_residuals.begin(),
                                    lio_diag.frame_diag_abs_residuals.end(), 0.0) / accepted_count
                  : std::numeric_limits<double>::quiet_NaN())
              << " ABS_P95=" << rolling_quantile(lio_diag.frame_diag_abs_residuals, 0.95)
              << " DELTA_POSITION_NORM=" << delta_position.norm()
              << " DELTA_VELOCITY_NORM=" << delta_velocity.norm()
              << " DELTA_ROTATION_DEG=" << lio_diag.frame_diag_delta_rotation_deg
              << std::endl;
    lio_diag.frame_diag_active = false;
}

void report_startup_group(const MeasureGroup &meas, double dt, bool update_ok,
                          std::uint64_t nearest_before,
                          std::uint64_t nearest_too_few_before,
                          std::uint64_t plane_before,
                          std::uint64_t residual_before,
                          const LioDiagStateSnapshot &ekf_pre_state,
                          const LioDiagStateSnapshot &ekf_propagated_state,
                          const LioDiagStateSnapshot &ekf_post_state)
{
    (void)dt;
    if (!init_map)
        return;

    Eigen::Vector3d pos, vel, gravity, acc_bias, gyro_bias;
    V3D euler_deg;
    get_lio_diag_state(pos, vel, gravity, acc_bias, gyro_bias, euler_deg);

    const std::uint64_t nearest_delta = lio_diag.nearest_reject - nearest_before;
    const std::uint64_t plane_delta = lio_diag.plane_reject - plane_before;
    const std::uint64_t residual_delta = lio_diag.residual_reject - residual_before;

    if (lio_diag.frame_diag_active)
    {
        if (!lio_diag.frame_diag_state_valid)
        {
            lio_diag.frame_diag_state_valid = true;
            lio_diag.frame_diag_pre_position = ekf_pre_state.pos;
            lio_diag.frame_diag_pre_velocity = ekf_pre_state.vel;
        }
        lio_diag.frame_diag_post_position = ekf_post_state.pos;
        lio_diag.frame_diag_post_velocity = ekf_post_state.vel;
        lio_diag.frame_diag_ekf_call_count++;
        lio_diag.frame_diag_ekf_success_any = lio_diag.frame_diag_ekf_success_any || update_ok;
        lio_diag.frame_diag_total_effect_num += static_cast<std::uint64_t>(std::max(lio_diag.effect_num_last, 0));
        lio_diag.frame_diag_total_accepted_points += lio_diag.current_accepted_point_count;
        lio_diag.frame_diag_total_nearest_reject += lio_diag.nearest_reject - nearest_before;
        lio_diag.frame_diag_total_plane_reject += plane_delta;
        lio_diag.frame_diag_total_residual_reject += residual_delta;
        lio_diag.frame_diag_delta_rotation_deg +=
            (ekf_post_state.euler_deg - ekf_propagated_state.euler_deg).norm();
        const auto &update_hth = use_imu_as_input
            ? kf_input.last_update_hth()
            : kf_output.last_update_hth();
        if (lio_diag.current_accepted_point_count > 0)
        {
            lio_diag.frame_diag_hth_sum += update_hth;
            for (const double residual : lio_diag.current_signed_residuals)
            {
                lio_diag.frame_diag_signed_residual_sum += residual;
                lio_diag.frame_diag_signed_residual_sq_sum += residual * residual;
            }
            lio_diag.frame_diag_abs_residuals.insert(
                lio_diag.frame_diag_abs_residuals.end(),
                lio_diag.current_abs_residuals.begin(),
                lio_diag.current_abs_residuals.end());
        }
    }
    const char *reason = "matched_or_other";
    if (nearest_delta > 0)
        reason = "nearest";
    else if (plane_delta > 0)
        reason = "plane";
    else if (residual_delta > 0)
        reason = "residual";
    else if (!update_ok)
        reason = "invalid";

    const std::size_t measure_group_imu_count = meas.sync_imu_count;
    const std::size_t imu_count = measure_group_imu_count > 0
        ? measure_group_imu_count
        : meas.sync_imu_queue_count;
    double imu_first = std::numeric_limits<double>::quiet_NaN();
    double imu_last = std::numeric_limits<double>::quiet_NaN();
    double imu_first_dt = std::numeric_limits<double>::quiet_NaN();
    double imu_last_dt = std::numeric_limits<double>::quiet_NaN();
    double imu_span = std::numeric_limits<double>::quiet_NaN();
    if (measure_group_imu_count > 0)
    {
        imu_first = meas.sync_imu_first_stamp;
        imu_last = meas.sync_imu_last_stamp;
    }
    else if (meas.sync_imu_queue_count > 0)
    {
        imu_first = meas.sync_imu_queue_first_stamp;
        imu_last = meas.sync_imu_queue_last_stamp;
    }
    if (std::isfinite(imu_first) && std::isfinite(imu_last))
    {
        imu_first_dt = imu_first - meas.lidar_beg_time;
        imu_last_dt = imu_last - meas.lidar_last_time;
        imu_span = imu_last - imu_first;
    }

    const double largest_gap = std::max(std::abs(imu_first_dt), std::abs(imu_last_dt));
    constexpr double large_gap_threshold_s = 0.01;
    if (imu_count == 0)
        lio_diag.startup_groups_with_zero_imu++;
    if (std::isfinite(largest_gap))
    {
        lio_diag.startup_max_imu_cloud_gap_s =
            std::max(lio_diag.startup_max_imu_cloud_gap_s, largest_gap);
        if (largest_gap > large_gap_threshold_s)
            lio_diag.startup_groups_with_large_imu_cloud_gap++;
    }
    if (update_ok && lio_diag.startup_first_success_group == std::numeric_limits<std::uint64_t>::max())
        lio_diag.startup_first_success_group = lio_diag.startup_group_count;

    LioDiagnosticCounters::RollingMeasurementGroup group;
    group.group_id = lio_diag.startup_group_count;
    group.lidar_begin_time = meas.lidar_beg_time;
    group.lidar_end_time = meas.lidar_last_time;
    group.cloud_point_count = meas.lidar ? meas.lidar->points.size() : 0;
    group.ekf_attempts = 1;
    group.ekf_success = update_ok ? 1 : 0;
    group.effect_num = lio_diag.effect_num_last;
    group.nearest_too_few = lio_diag.nearest_too_few_count - nearest_too_few_before;
    group.plane_reject = plane_delta;
    group.residual_reject = residual_delta;
    group.accepted_point_count = lio_diag.current_accepted_point_count;
    if (group.accepted_point_count > 0)
    {
        group.accepted_signed_residual_mean =
            lio_diag.current_signed_residual_sum / static_cast<double>(group.accepted_point_count);
        group.accepted_abs_residual_mean =
            lio_diag.current_abs_residual_sum / static_cast<double>(group.accepted_point_count);
        group.accepted_abs_residual_p50 = rolling_quantile(lio_diag.current_abs_residuals, 0.50);
        group.accepted_abs_residual_p95 = rolling_quantile(lio_diag.current_abs_residuals, 0.95);
        group.accepted_abs_residual_max = lio_diag.current_abs_residual_max;
        group.normal_mean = lio_diag.current_normal_sum / static_cast<double>(group.accepted_point_count);
        group.normal_abs_mean = lio_diag.current_normal_abs_sum / static_cast<double>(group.accepted_point_count);
    }
    group.normal_x_dominant_count = lio_diag.current_normal_x_dominant_count;
    group.normal_y_dominant_count = lio_diag.current_normal_y_dominant_count;
    group.normal_z_dominant_count = lio_diag.current_normal_z_dominant_count;
    group.pre_position = ekf_propagated_state.pos;
    group.post_position = ekf_post_state.pos;
    group.pre_velocity = ekf_propagated_state.vel;
    group.post_velocity = ekf_post_state.vel;
    group.delta_position_norm = (group.post_position - group.pre_position).norm();
    group.delta_rotation_deg = (ekf_post_state.euler_deg - ekf_propagated_state.euler_deg).norm();
    group.delta_velocity_norm = (group.post_velocity - group.pre_velocity).norm();
    group.delta_gyro_bias_norm = (ekf_post_state.gyro_bias - ekf_propagated_state.gyro_bias).norm();
    group.delta_acc_bias_norm = (ekf_post_state.acc_bias - ekf_propagated_state.acc_bias).norm();

    report_lidar_information_update(group, update_ok);

    if (lio_diag.rolling_groups.size() >= 20)
        lio_diag.rolling_groups.pop_front();
    lio_diag.rolling_groups.push_back(group);

    if (lio_diag.startup_group_count < 30)
    {
    std::cout << "[LIO-DIAG] STARTUP_GROUP"
              << " GROUP_ID=" << lio_diag.startup_group_count
              << " CLOUD_HEADER_STAMP=" << meas.lidar_beg_time
              << " CLOUD_RECEIVE_ORDER=" << meas.cloud_receive_order
              << " CLOUD_POINT_COUNT=" << meas.lidar->points.size()
              << " IMU_COUNT_IN_GROUP=" << imu_count
              << " MEASURE_GROUP_IMU_COUNT=" << measure_group_imu_count
              << " IMU_FIRST_HEADER_STAMP=" << imu_first
              << " IMU_LAST_HEADER_STAMP=" << imu_last
              << " IMU_FIRST_TO_CLOUD_DT=" << imu_first_dt
              << " IMU_LAST_TO_CLOUD_DT=" << imu_last_dt
              << " IMU_FIRST_TO_LIDAR_BEGIN_DT=" << imu_first_dt
              << " IMU_LAST_TO_LIDAR_END_DT=" << imu_last_dt
              << " IMU_SPAN_S=" << imu_span
              << " SYNC_QUEUE_CLOUD_SIZE_BEFORE=" << meas.sync_queue_cloud_size_before
              << " SYNC_QUEUE_IMU_SIZE_BEFORE=" << meas.sync_queue_imu_size_before
              << " CLOUD_DROPPED_BEFORE_GROUP=" << meas.cloud_dropped_before_group
              << " IMU_DROPPED_BEFORE_GROUP=" << meas.imu_dropped_before_group
              << " GROUP_FORM_REASON=" << meas.group_form_reason
              << " LIDAR_BEGIN_TIME=" << meas.lidar_beg_time
              << " LIDAR_END_TIME=" << meas.lidar_last_time
              << " EKF_ATTEMPTED=YES"
              << " EKF_SUCCESS=" << (update_ok ? "YES" : "NO")
              << " EFFECT_NUM=" << lio_diag.effect_num_last
              << " NEAREST_REJECT=" << nearest_delta
              << " PLANE_REJECT=" << plane_delta
              << " RESIDUAL_REJECT=" << residual_delta
              << " POS_AFTER=" << pos.transpose()
              << " VEL_AFTER=" << vel.transpose()
              << " ROLL_PITCH_YAW_AFTER=" << euler_deg.transpose()
              << " ACC_BIAS_AFTER=" << acc_bias.transpose()
              << " GYRO_BIAS_AFTER=" << gyro_bias.transpose()
              << " REJECT_REASON=" << reason
              << std::endl;
    }

    lio_diag.startup_group_count++;
    const double drift = (pos - lio_diag.bias_diag_map_init_pos).norm();
    if (!lio_diag.drift_0p2_reported && drift >= 0.2)
    {
        lio_diag.drift_0p2_reported = true;
        report_rolling_context("[LIO-DIAG] DRIFT_0P2_CONTEXT");
    }
    if (lio_diag.startup_group_count == 30)
        report_startup_group_summary();
}

void report_lio_diagnostics_if_due()
{
    if (flg_exit)
    {
        report_startup_group_summary();
        report_stable_exit_context();
    }
    const double now = omp_get_wtime();
    if (now - lio_diag_last_report_time < 1.0)
        return;
    lio_diag_last_report_time = now;

    const double undistort_mean = lio_diag.undistort_samples == 0
        ? 0.0
        : static_cast<double>(lio_diag.undistort_sum) / lio_diag.undistort_samples;
    const double down_mean = lio_diag.down_samples == 0
        ? 0.0
        : static_cast<double>(lio_diag.down_sum) / lio_diag.down_samples;
    const double effect_mean = lio_diag.effect_num_samples == 0
        ? 0.0
        : static_cast<double>(lio_diag.effect_num_sum) / lio_diag.effect_num_samples;
    const int map_points = init_map ? ikdtree.validnum() : init_feats_world->size();

    Eigen::Vector3d state_pos, state_vel, state_gravity, state_acc_bias, state_gyro_bias;
    V3D state_euler_deg;
    get_lio_diag_state(state_pos, state_vel, state_gravity, state_acc_bias, state_gyro_bias, state_euler_deg);
    if (init_map && !lio_diag_state_init_valid)
    {
        lio_diag_state_pos_init = state_pos;
        lio_diag_state_init_valid = true;
    }
    const double state_position_drift = lio_diag_state_init_valid
        ? (state_pos - lio_diag_state_pos_init).norm()
        : std::numeric_limits<double>::quiet_NaN();

    std::vector<double> nearest_distances = lio_diag.nearest_success_distances;
    std::sort(nearest_distances.begin(), nearest_distances.end());
    const auto nearest_quantile = [&nearest_distances](double q) {
        if (nearest_distances.empty())
            return std::numeric_limits<double>::quiet_NaN();
        const std::size_t index = static_cast<std::size_t>(q * (nearest_distances.size() - 1));
        return nearest_distances[index];
    };
    double nearest_mean = 0.0;
    for (const double distance : nearest_distances)
        nearest_mean += distance;
    if (!nearest_distances.empty())
        nearest_mean /= nearest_distances.size();
    else
        nearest_mean = std::numeric_limits<double>::quiet_NaN();

    std::cout << "[LIO-DIAG]"
              << " frames=" << lio_diag.frame_count
              << " sync_ok=" << lio_diag.sync_package_ok
              << " sync_fail=" << lio_diag.sync_package_fail
              << " imu_process_ok=" << lio_diag.imu_process_count
              << " imu_process_count=" << lio_diag.imu_process_count
              << " feats_undistort_mean=" << undistort_mean
              << " feats_down_mean=" << down_mean
              << " map_initialized=" << (init_map ? "YES" : "NO")
              << " map_points=" << map_points
              << " time_groups=" << lio_diag.time_groups_last
              << " ekf_attempts=" << lio_diag.ekf_update_attempts
              << " ekf_success=" << lio_diag.ekf_update_success
              << " ekf_fail=" << lio_diag.ekf_update_fail
              << " effect_num=" << lio_diag.effect_num_last
              << " effect_num_last=" << lio_diag.effect_num_last
              << " effect_num_mean=" << effect_mean
              << " effect_num_zero_count=" << lio_diag.effect_num_zero_count
              << " nearest_query_count=" << lio_diag.nearest_query_count
              << " nearest_reject=" << lio_diag.nearest_reject
              << " nearest_too_few=" << lio_diag.nearest_too_few_count
              << " nearest_too_far=" << lio_diag.nearest_too_far_count
              << " nearest_other_reject=" << lio_diag.nearest_other_reject_count
              << " plane_reject=" << lio_diag.plane_reject
              << " residual_reject=" << lio_diag.residual_reject
              << " invalid_reject=" << lio_diag.effect_num_zero_count
              << " other_invalid_count=" << 0
              << " odom_publish=" << lio_diag.odom_publish_count
              << " tf_publish=" << lio_diag.tf_publish_count
              << " registered_cloud_publish=" << lio_diag.registered_cloud_publish_count
              << std::endl;

    std::cout << "[LIO-DIAG] NEAREST_DISTANCE"
              << " min=" << (nearest_distances.empty() ? std::numeric_limits<double>::quiet_NaN() : nearest_distances.front())
              << " mean=" << nearest_mean
              << " p50=" << nearest_quantile(0.50)
              << " p95=" << nearest_quantile(0.95)
              << " max=" << (nearest_distances.empty() ? std::numeric_limits<double>::quiet_NaN() : nearest_distances.back())
              << std::endl;

    std::cout << "[LIO-DIAG] STATE"
              << " STATE_POS_X=" << state_pos.x()
              << " STATE_POS_Y=" << state_pos.y()
              << " STATE_POS_Z=" << state_pos.z()
              << " STATE_VEL_X=" << state_vel.x()
              << " STATE_VEL_Y=" << state_vel.y()
              << " STATE_VEL_Z=" << state_vel.z()
              << " STATE_ROLL_DEG=" << state_euler_deg(0)
              << " STATE_PITCH_DEG=" << state_euler_deg(1)
              << " STATE_YAW_DEG=" << state_euler_deg(2)
              << " STATE_GRAV_X=" << state_gravity.x()
              << " STATE_GRAV_Y=" << state_gravity.y()
              << " STATE_GRAV_Z=" << state_gravity.z()
              << " STATE_ACC_BIAS_X=" << state_acc_bias.x()
              << " STATE_ACC_BIAS_Y=" << state_acc_bias.y()
              << " STATE_ACC_BIAS_Z=" << state_acc_bias.z()
              << " STATE_GYRO_BIAS_X=" << state_gyro_bias.x()
              << " STATE_GYRO_BIAS_Y=" << state_gyro_bias.y()
              << " STATE_GYRO_BIAS_Z=" << state_gyro_bias.z()
              << " STATE_POSITION_DRIFT_FROM_INIT_M=" << state_position_drift
              << " STATE_VELOCITY_NORM=" << state_vel.norm()
              << " IMU_ACC_MEAN_X=" << p_imu->mean_acc.x()
              << " IMU_ACC_MEAN_Y=" << p_imu->mean_acc.y()
              << " IMU_ACC_MEAN_Z=" << p_imu->mean_acc.z()
              << " IMU_ACC_NORM=" << p_imu->mean_acc.norm()
              << " IMU_GYRO_MEAN_X=" << p_imu->mean_gyr_value().x()
              << " IMU_GYRO_MEAN_Y=" << p_imu->mean_gyr_value().y()
              << " IMU_GYRO_MEAN_Z=" << p_imu->mean_gyr_value().z()
              << std::endl;

    report_bias_source_diagnostics();
    report_bias_at_drift_diagnostics();
    report_startup_input_diagnostics();

    std::cout << "[LIO-DIAG] EKF_CUMULATIVE_FIRST200"
              << " TRACE_COUNT=" << lio_diag_trace_count
              << " CUM_PROP_VEL_CHANGE=" << lio_diag_cum_prop_vel.transpose()
              << " CUM_PROP_VEL_CHANGE_NORM_SUM=" << lio_diag_cum_prop_vel_norm
              << " CUM_EKF_VEL_CHANGE=" << lio_diag_cum_ekf_vel.transpose()
              << " CUM_EKF_VEL_CHANGE_NORM_SUM=" << lio_diag_cum_ekf_vel_norm
              << " CUM_PROP_ROT_CHANGE_DEG=" << lio_diag_cum_prop_rot.transpose()
              << " CUM_PROP_ROT_CHANGE_NORM_SUM=" << lio_diag_cum_prop_rot_norm
              << " CUM_EKF_ROT_CHANGE_DEG=" << lio_diag_cum_ekf_rot.transpose()
              << " CUM_EKF_ROT_CHANGE_NORM_SUM=" << lio_diag_cum_ekf_rot_norm
              << " CUM_EKF_ACC_BIAS_CHANGE=" << lio_diag_cum_ekf_acc_bias.transpose()
              << " CUM_EKF_GYRO_BIAS_CHANGE=" << lio_diag_cum_ekf_gyro_bias.transpose()
              << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> plane_solver(lio_diag.plane_normal_outer_sum);
    const Eigen::Vector3d plane_eigenvalues_ascending = plane_solver.eigenvalues();
    const double lambda1 = plane_eigenvalues_ascending(2);
    const double lambda2 = plane_eigenvalues_ascending(1);
    const double lambda3 = plane_eigenvalues_ascending(0);
    const double plane_condition_ratio = lambda1 > 0.0
        ? lambda3 / lambda1
        : std::numeric_limits<double>::quiet_NaN();
    const double normal_count = static_cast<double>(lio_diag.plane_normal_count);
    std::cout << "[LIO-DIAG] PLANE_NORMAL_EIGEN"
              << " PLANE_NORMAL_EIGENVALUES=" << lambda1 << "," << lambda2 << "," << lambda3
              << " PLANE_NORMAL_CONDITION_RATIO=" << plane_condition_ratio
              << " NORMAL_Z_DOMINANT_RATIO=" << (normal_count > 0.0 ? lio_diag.normal_z_dominant_count / normal_count : 0.0)
              << " NORMAL_X_DOMINANT_RATIO=" << (normal_count > 0.0 ? lio_diag.normal_x_dominant_count / normal_count : 0.0)
              << " NORMAL_Y_DOMINANT_RATIO=" << (normal_count > 0.0 ? lio_diag.normal_y_dominant_count / normal_count : 0.0)
              << " PLANE_NORMAL_COUNT=" << lio_diag.plane_normal_count
              << std::endl;

    std::cout << "[LIO-DIAG] SPACE";
    if (init_map)
    {
        const BoxPointType map_box = ikdtree.tree_range();
        const Eigen::Vector3d map_min(map_box.vertex_min[0], map_box.vertex_min[1], map_box.vertex_min[2]);
        const Eigen::Vector3d map_max(map_box.vertex_max[0], map_box.vertex_max[1], map_box.vertex_max[2]);
        const Eigen::Vector3d map_center = 0.5 * (map_min + map_max);
        std::cout << " MAP_BBOX_X_MIN=" << map_min.x()
                  << " MAP_BBOX_X_MAX=" << map_max.x()
                  << " MAP_BBOX_Y_MIN=" << map_min.y()
                  << " MAP_BBOX_Y_MAX=" << map_max.y()
                  << " MAP_BBOX_Z_MIN=" << map_min.z()
                  << " MAP_BBOX_Z_MAX=" << map_max.z()
                  << " MAP_CENTER_X=" << map_center.x()
                  << " MAP_CENTER_Y=" << map_center.y()
                  << " MAP_CENTER_Z=" << map_center.z();
        if (lio_diag_scan_bbox_valid)
        {
            const Eigen::Vector3d scan_center = 0.5 * (lio_diag_scan_bbox_min + lio_diag_scan_bbox_max);
            std::cout << " CURRENT_SCAN_WORLD_BBOX_X_MIN=" << lio_diag_scan_bbox_min.x()
                      << " CURRENT_SCAN_WORLD_BBOX_X_MAX=" << lio_diag_scan_bbox_max.x()
                      << " CURRENT_SCAN_WORLD_BBOX_Y_MIN=" << lio_diag_scan_bbox_min.y()
                      << " CURRENT_SCAN_WORLD_BBOX_Y_MAX=" << lio_diag_scan_bbox_max.y()
                      << " CURRENT_SCAN_WORLD_BBOX_Z_MIN=" << lio_diag_scan_bbox_min.z()
                      << " CURRENT_SCAN_WORLD_BBOX_Z_MAX=" << lio_diag_scan_bbox_max.z()
                      << " SCAN_WORLD_CENTER_X=" << scan_center.x()
                      << " SCAN_WORLD_CENTER_Y=" << scan_center.y()
                      << " SCAN_WORLD_CENTER_Z=" << scan_center.z()
                      << " MAP_SCAN_CENTER_DISTANCE_M=" << (map_center - scan_center).norm();
        }
        else
        {
            std::cout << " CURRENT_SCAN_WORLD_BBOX=UNAVAILABLE";
        }
    }
    else
    {
        std::cout << " MAP_BBOX=UNAVAILABLE CURRENT_SCAN_WORLD_BBOX=UNAVAILABLE";
    }
    std::cout << std::endl;
    report_geometry_retention_diagnostics();
    report_early_geometry_diagnostics();
    report_early_ekf_gain_diagnostics();
}

void SigHandle(int sig)
{
    flg_exit = true;
    printf("catch sig %d", sig);
    sig_buffer.notify_all();
}


inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang;
    if (!use_imu_as_input)
    {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    }
    else
    {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }

    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
    if (use_imu_as_input)
    {

        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2));
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2));
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2));
    }
    else
    {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2));
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2));
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2));
    }
    fprintf(fp, "\r\n");
    fflush(fp);
}


void pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{

    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            p_body_imu = kf_output.x_.offset_R_L_I.normalized() * p_body_lidar + kf_output.x_.offset_T_L_I;
        }
        else
        {
            p_body_imu = kf_input.x_.offset_R_L_I.normalized() * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    }
    else
    {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);

    po->intensity = pi->intensity;
}

int points_cache_size = 0;
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
}


BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.shrink_to_fit();

    V3D pos_LiD;
    if (use_imu_as_input)
    {
        pos_LiD = kf_input.x_.pos + kf_input.x_.rot.normalized() * Lidar_T_wrt_IMU;
    }
    else
    {
        pos_LiD = kf_output.x_.pos + kf_output.x_.rot.normalized() * Lidar_T_wrt_IMU;
    }

    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }

    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    if (cub_needrm.size() > 0)
        int kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}


void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    // std::cout << "standard_pcl_cbk() run once!\n";

    mtx_buffer.lock();

    scan_count++;
    lio_diag.frame_count++;

    double preprocess_start_time = omp_get_wtime();

    if (get_time_in_sec(msg->header.stamp) < last_timestamp_lidar)
    {
        printf("lidar loop back, clear buffer");
        lidar_drop_count++;

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = get_time_in_sec(msg->header.stamp);
    const std::uint64_t receive_order = ++lidar_receive_order;

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());

    double time_div = get_time_in_sec(msg->header.stamp);

    p_pre->process(msg, ptr);

    record_startup_cloud_diagnostic(get_time_in_sec(msg->header.stamp), *ptr);

    // std::cout << "ptr->points.size() = " << ptr->points.size() << std::endl;

    if (cut_frame)
    {

        sort(ptr->points.begin(), ptr->points.end(), time_list);

        for (int i = 0; i < ptr->size(); i++)
        {

            ptr_div->push_back(ptr->points[i]);

            if (ptr->points[i].curvature / double(1000) + get_time_in_sec(msg->header.stamp) - time_div > cut_frame_time_interval)
            {
                if (ptr_div->size() < 1)
                    continue;

                PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
                *ptr_div_i = *ptr_div;

                lidar_buffer.push_back(ptr_div_i);

                time_buffer.push_back(time_div);
                lidar_receive_order_buffer.push_back(receive_order);
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }

        if (!ptr_div->empty())
        {
            lidar_buffer.push_back(ptr_div);

            time_buffer.push_back(time_div);
            lidar_receive_order_buffer.push_back(receive_order);
        }
    }
    else if (con_frame)
    {

        if (frame_ct == 0)
        {
            time_con = last_timestamp_lidar;
        }

        if (frame_ct < con_frame_num)
        {
            for (int i = 0; i < ptr->size(); i++)
            {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct++;
        }

        else
        {
            PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
            *ptr_con_i = *ptr_con;
            lidar_buffer.push_back(ptr_con_i);
            double time_con_i = time_con;
            time_buffer.push_back(time_con_i);
            lidar_receive_order_buffer.push_back(receive_order);
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else
    {
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(get_time_in_sec(msg->header.stamp));
        lidar_receive_order_buffer.push_back(receive_order);
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


// void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
// {

//     mtx_buffer.lock();

//     double preprocess_start_time = omp_get_wtime();

//     scan_count++;

//     if (get_time_in_sec(msg->header.stamp) < last_timestamp_lidar)
//     {
//         ROS_ERROR("lidar loop back, clear buffer");

//         mtx_buffer.unlock();
//         sig_buffer.notify_all();
//         return;
//     }

//     last_timestamp_lidar = get_time_in_sec(msg->header.stamp);

//     PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
//     PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());

//     p_pre->process(msg, ptr);
//     double time_div = get_time_in_sec(msg->header.stamp);

//     if (cut_frame)
//     {

//         sort(ptr->points.begin(), ptr->points.end(), time_list);

//         for (int i = 0; i < ptr->size(); i++)
//         {

//             ptr_div->push_back(ptr->points[i]);
//             if (ptr->points[i].curvature / double(1000) + get_time_in_sec(msg->header.stamp) - time_div > cut_frame_time_interval)
//             {
//                 if (ptr_div->size() < 1)
//                     continue;
//                 PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());

//                 *ptr_div_i = *ptr_div;

//                 lidar_buffer.push_back(ptr_div_i);
//                 time_buffer.push_back(time_div);
//                 time_div += ptr->points[i].curvature / double(1000);
//                 ptr_div->clear();
//             }
//         }
//         if (!ptr_div->empty())
//         {
//             lidar_buffer.push_back(ptr_div);

//             time_buffer.push_back(time_div);
//         }
//     }
//     else if (con_frame)
//     {
//         if (frame_ct == 0)
//         {

//             time_con = last_timestamp_lidar;
//         }
//         if (frame_ct < con_frame_num)
//         {
//             for (int i = 0; i < ptr->size(); i++)
//             {
//                 ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
//                 ptr_con->push_back(ptr->points[i]);
//             }
//             frame_ct++;
//         }
//         else
//         {
//             PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
//             *ptr_con_i = *ptr_con;
//             double time_con_i = time_con;
//             lidar_buffer.push_back(ptr_con_i);
//             time_buffer.push_back(time_con_i);
//             ptr_con->clear();
//             frame_ct = 0;
//         }
//     }
//     else
//     {
//         lidar_buffer.emplace_back(ptr);
//         time_buffer.emplace_back(get_time_in_sec(msg->header.stamp));
//     }
//     s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
//     mtx_buffer.unlock();
//     sig_buffer.notify_all();
// }

void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr msg_in)
{

    publish_count++;

    sensor_msgs::msg::Imu::Ptr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_in_sec(msg_in->header.stamp) - time_lag_imu_to_lidar);

    double timestamp = get_time_in_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        printf("imu loop back, clear deque");
        imu_drop_count++;

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    imu_deque.emplace_back(msg);

    record_startup_imu_diagnostic(timestamp, *msg);

    last_timestamp_imu = timestamp;

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


bool sync_packages(MeasureGroup &meas)
{

    if (!imu_en)
    {
        if (!lidar_buffer.empty())
        {

            meas.lidar = lidar_buffer.front();
            meas.cloud_receive_order = lidar_receive_order_buffer.front();
            meas.sync_queue_cloud_size_before = lidar_buffer.size();
            meas.sync_queue_imu_size_before = imu_deque.size();
            meas.cloud_dropped_before_group = lidar_drop_count;
            meas.imu_dropped_before_group = imu_drop_count;
            meas.group_form_reason = "no_imu_mode";
            meas.sync_imu_count = 0;
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_buffer.pop_front();
            lidar_receive_order_buffer.pop_front();

            if (meas.lidar->points.size() < 1)
            {
                cout << "lose lidar" << std::endl;
                return false;
            }

            double end_time = meas.lidar->points.back().curvature;
            for (auto pt : meas.lidar->points)
            {
                if (pt.curvature > end_time)
                {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);

            meas.lidar_last_time = lidar_end_time;

            if (lio_diag.startup_input_started &&
                meas.lidar_beg_time <= lio_diag.startup_first_cloud_time + 10.0)
                lio_diag.startup_sync_success_count++;

            begin_lidar_frame_diagnostic(meas);
            return true;
        }
        return false;
    }

    if (lidar_buffer.empty() || imu_deque.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {

        meas.lidar = lidar_buffer.front();
        meas.cloud_receive_order = lidar_receive_order_buffer.front();
        meas.sync_queue_cloud_size_before = lidar_buffer.size();
        meas.sync_queue_imu_size_before = imu_deque.size();
        meas.cloud_dropped_before_group = lidar_drop_count;
        meas.imu_dropped_before_group = imu_drop_count;
        meas.group_form_reason = "waiting_for_imu_header_time_coverage";
        meas.sync_imu_count = 0;

        if (meas.lidar->points.size() < 1)
        {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            lidar_receive_order_buffer.pop_front();
            return false;
        }

        meas.lidar_beg_time = time_buffer.front();

        double end_time = meas.lidar->points.back().curvature;
        for (auto pt : meas.lidar->points)
        {
            if (pt.curvature > end_time)
            {
                end_time = pt.curvature;
            }
        }
        lidar_end_time = meas.lidar_beg_time + end_time / double(1000);

        meas.lidar_last_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_)
    {
        double imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time)
                break;
            meas.imu.emplace_back(imu_deque.front());
            const double current_imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
            if (meas.sync_imu_count == 0)
                meas.sync_imu_first_stamp = current_imu_time;
            meas.sync_imu_last_stamp = current_imu_time;
            meas.sync_imu_count++;
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }
    else if (!init_map)
    {
        double imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        meas.imu.emplace_back(imu_last_ptr);
        meas.sync_imu_count = 1;
        meas.sync_imu_first_stamp = get_time_in_sec(imu_last_ptr->header.stamp);
        meas.sync_imu_last_stamp = meas.sync_imu_first_stamp;

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time)
                break;
            meas.imu.emplace_back(imu_deque.front());
            const double current_imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
            if (meas.sync_imu_count == 0)
                meas.sync_imu_first_stamp = current_imu_time;
            meas.sync_imu_last_stamp = current_imu_time;
            meas.sync_imu_count++;
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_receive_order_buffer.pop_front();
    lidar_pushed = false;
    meas.group_form_reason = "imu_header_time_covered_lidar_scan_end";
    meas.sync_imu_queue_count = 0;
    for (const auto &imu_msg : imu_deque)
    {
        const double imu_stamp = get_time_in_sec(imu_msg->header.stamp);
        if (imu_stamp > meas.lidar_last_time)
            break;
        if (meas.sync_imu_queue_count == 0)
            meas.sync_imu_queue_first_stamp = imu_stamp;
        meas.sync_imu_queue_last_stamp = imu_stamp;
        meas.sync_imu_queue_count++;
    }
    if (lio_diag.startup_input_started &&
        meas.lidar_beg_time <= lio_diag.startup_first_cloud_time + 10.0)
        lio_diag.startup_sync_success_count++;
    begin_lidar_frame_diagnostic(meas);
    return true;
}


int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++)
    {
        if (!Nearest_Points[i].empty())
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            /* If the nearest points is definitely outside the downsample box */
            if (fabs(points_near[0].x - mid_point.x) > 1.732 * filter_size_map_min || fabs(points_near[0].y - mid_point.y) > 1.732 * filter_size_map_min || fabs(points_near[0].z - mid_point.z) > 1.732 * filter_size_map_min)
            {
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                continue;
            }
            /* Check if there is a point already in the downsample box */
            float dist = calc_dist<float>(feats_down_world->points[i], mid_point);
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++)
            {
                /* Those points which are outside the downsample box should not be considered. */
                if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].y - mid_point.y) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].z - mid_point.z) < 0.5 * filter_size_map_min)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.emplace_back(feats_down_world->points[i]);
        }
        else
        {

            PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
        }
    }
    int add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

void publish_init_kdtree(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    int size_init_ikdtree = ikdtree.size();
    PointCloudXYZI::Ptr laserCloudInit(new PointCloudXYZI(size_init_ikdtree, 1));

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    PointVector().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);

    laserCloudInit->points = ikdtree.PCL_Storage;
    pcl::toROSMsg(*laserCloudInit, laserCloudmsg);

    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes->publish(laserCloudmsg);
}


PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {

            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes->publish(laserCloudmsg);
        lio_diag.registered_cloud_publish_count++;
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}


void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        pointBodyLidarToIMU(&feats_undistort->points[i],
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

template <typename T>
void set_posestamp(T &out)
{
    if (!use_imu_as_input)
    {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        out.orientation.x = kf_output.x_.rot.coeffs()[0];
        out.orientation.y = kf_output.x_.rot.coeffs()[1];
        out.orientation.z = kf_output.x_.rot.coeffs()[2];
        out.orientation.w = kf_output.x_.rot.coeffs()[3];
    }
    else
    {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        out.orientation.x = kf_input.x_.rot.coeffs()[0];
        out.orientation.y = kf_input.x_.rot.coeffs()[1];
        out.orientation.z = kf_input.x_.rot.coeffs()[2];
        out.orientation.w = kf_input.x_.rot.coeffs()[3];
    }
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = get_ros_time(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);

    pubOdomAftMapped->publish(odomAftMapped);
    lio_diag.odom_publish_count++;

    // tf2::Transform transform;
    // tf2::Quaternion q;
    // transform.setOrigin(tf2::Vector3(odomAftMapped.pose.pose.position.x,
    //                                 odomAftMapped.pose.pose.position.y,
    //                                 odomAftMapped.pose.pose.position.z));
    // q.setW(odomAftMapped.pose.pose.orientation.w);
    // q.setX(odomAftMapped.pose.pose.orientation.x);
    // q.setY(odomAftMapped.pose.pose.orientation.y);
    // q.setZ(odomAftMapped.pose.pose.orientation.z);
    // transform.setRotation(q);

    geometry_msgs::msg::TransformStamped trans_odom_to_base;
    trans_odom_to_base.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans_odom_to_base.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans_odom_to_base.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans_odom_to_base.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans_odom_to_base.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans_odom_to_base.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans_odom_to_base.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    trans_odom_to_base.header = odomAftMapped.header;
    trans_odom_to_base.child_frame_id = odomAftMapped.child_frame_id;
    tf_br->sendTransform(trans_odom_to_base);
    lio_diag.tf_publish_count++;
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose.pose);

    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;

    {
        path.poses.emplace_back(msg_body_pose);
        pubPath->publish(path);
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("laserMapping");

    readParameters(node);
    cout << "lidar_type: " << lidar_type << endl << flush;

    path.header.stamp = get_ros_time(lidar_end_time);
    path.header.frame_id = "camera_init";

    int frame_num = 0;
    double aver_time_consu = 0,
           aver_time_icp = 0,
           aver_time_match = 0,
           aver_time_incre = 0,
           aver_time_solve = 0,
           aver_time_propag = 0;
    std::time_t startTime, endTime;

    double FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    double HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    memset(point_selected_surf, true, sizeof(point_selected_surf));

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

    if (extrinsic_est_en)
    {

        if (!use_imu_as_input)
        {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }

        else
        {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }

    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    kf_input.init_dyn_share_modified(get_f_input, df_dx_input, h_model_input);
    kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);

    Eigen::Matrix<double, 24, 24> P_init = MD(24, 24)::Identity() * 0.01;
    P_init.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6, 6)::Identity() * 0.001;
    P_init.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
    kf_input.change_P(P_init);

    Eigen::Matrix<double, 30, 30> P_init_output = MD(30, 30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
    P_init_output.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
    kf_input.change_P(P_init);
    kf_output.change_P(P_init_output);

    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();

    tf_br = std::make_unique<tf2_ros::TransformBroadcaster>(*node);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_out, fout_imu_pbp;
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"), ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    
    auto sub_pcl = node->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, 2000, standard_pcl_cbk);

    auto sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 2000, imu_cbk);

    auto pubLaserCloudFullRes = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);

    auto pubLaserCloudFullRes_body = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 1000);

    auto pubLaserCloudEffect = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 1000);

    auto pubLaserCloudMap = node->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 1000);

    auto pubOdomAftMapped = node->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init", 1000);

    auto pubPath = node->create_publisher<nav_msgs::msg::Path>("/path", 1000);

    auto plane_pub = node->create_publisher<visualization_msgs::msg::Marker>("/planner_normal", 1000);

    signal(SIGINT, SigHandle);

    rclcpp::Rate rate(5000);
    while (rclcpp::ok())
    {

        if (flg_exit)
            break;

        rclcpp::spin_some(node);
        report_lio_diagnostics_if_due();

        if (sync_packages(Measures) == false)
        {
            lio_diag.sync_package_fail++;
            rate.sleep();
            continue;
        }
        lio_diag.sync_package_ok++;

        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            flg_first_scan = false;
            cout << "first lidar time" << first_lidar_time << endl;
        }

        if (flg_reset)
        {
            printf("reset when rosbag play back");
            p_imu->Reset();
            flg_reset = false;
            continue;
        }

        double t0, t1, t2, t3, t4, t5, match_start, solve_start;
        match_time = 0;
        solve_time = 0;
        propag_time = 0;
        update_time = 0;
        t0 = omp_get_wtime();

        lio_diag.imu_process_count++;
        p_imu->Process(Measures, feats_undistort);
        lio_diag.undistort_samples++;
        lio_diag.undistort_sum += feats_undistort->points.size();

        if (feats_undistort->empty() || feats_undistort == NULL)
        {
            continue;
        }

        if (imu_en)
        {
            if (!p_imu->gravity_align_)
            {
                while (Measures.lidar_beg_time > get_time_in_sec(imu_next.header.stamp))
                {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    imu_deque.pop_front();
                }
                if (non_station_start)
                {
                    state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc *= -1;
                }
                else
                {
                    state_in.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                    state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                    state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                }
                if (gravity_align)
                {
                    Eigen::Matrix3d rot_init;
                    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                    p_imu->Set_init(state_in.gravity, rot_init);
                    state_in.gravity = state_out.gravity = p_imu->gravity_;
                    state_in.rot = state_out.rot = rot_init;
                    state_in.rot.normalize();
                    state_out.rot.normalize();
                    state_out.acc = -rot_init.transpose() * state_out.gravity;
                }
                const LioDiagStateSnapshot input_before_init = snapshot_lio_state(kf_input.x_);
                kf_input.change_x(state_in);
                if (use_imu_as_input)
                    record_bias_init_reset(input_before_init, snapshot_lio_state(kf_input.x_));

                const LioDiagStateSnapshot output_before_init = snapshot_lio_state(kf_output.x_);
                kf_output.change_x(state_out);
                if (!use_imu_as_input)
                    record_bias_init_reset(output_before_init, snapshot_lio_state(kf_output.x_));
            }
        }
        else
        {
            if (!p_imu->gravity_align_)
            {
                state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                state_out.acc << VEC_FROM_ARRAY(gravity_init);
                state_out.acc *= -1;
            }
        }

        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        t1 = omp_get_wtime();
        if (space_down_sample)
        {
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        }
        else
        {
            feats_down_body = Measures.lidar;

            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        }
        time_seq = time_compressing<int>(feats_down_body);
        feats_down_size = feats_down_body->points.size();
        lio_diag.down_samples++;
        lio_diag.down_sum += feats_down_size;
        lio_diag.time_groups_last = time_seq.size();
        record_prematch_scan_geometry();

        /*** initialize the map kdtree ***/
        if (!init_map)
        {
            if (ikdtree.Root_Node == nullptr)

            {
                ikdtree.set_downsample_param(filter_size_map_min);
            }

            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++)
            {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            update_lio_diag_scan_bbox();

            for (size_t i = 0; i < feats_down_world->size(); i++)
            {
                init_feats_world->points.emplace_back(feats_down_world->points[i]);
            }

            if (init_feats_world->size() < init_map_size)
                continue;

            ikdtree.Build(init_feats_world->points);
            init_map = true;

            if (!lio_diag.map_initialized_reported)
            {
                lio_diag.map_initialized_reported = true;
                if (lio_diag.startup_input_started)
                {
                    lio_diag.startup_map_initialized_time = Measures.lidar_beg_time;
                    lio_diag.startup_initial_map_points = ikdtree.validnum();
                }
                lio_diag.bias_diag_init_valid = true;
                if (use_imu_as_input)
                {
                    lio_diag.bias_diag_init_gyro = kf_input.x_.bg;
                    lio_diag.bias_diag_init_acc = kf_input.x_.ba;
                }
                else
                {
                    lio_diag.bias_diag_init_gyro = kf_output.x_.bg;
                    lio_diag.bias_diag_init_acc = kf_output.x_.ba;
                }
                lio_diag.bias_diag_map_init_pos = use_imu_as_input
                    ? kf_input.x_.pos
                    : kf_output.x_.pos;
                lio_diag.bias_diag_frame_count = 0;
                std::cout << "[LIO-DIAG] MAP_INITIALIZED"
                          << " map_points=" << ikdtree.validnum()
                          << " feats_down=" << feats_down_size
                          << std::endl;
            }

            publish_init_kdtree(pubLaserCloudMap);
            continue;
        }

        /*** ICP and Kalman filter update ***/

        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);

        Nearest_Points.resize(feats_down_size);

        t2 = omp_get_wtime();

        /*** iterated state estimation ***/

        crossmat_list.reserve(feats_down_size);
        pbody_list.reserve(feats_down_size);

        for (size_t i = 0; i < feats_down_body->size(); i++)
        {

            V3D point_this(feats_down_body->points[i].x,
                           feats_down_body->points[i].y,
                           feats_down_body->points[i].z);
            pbody_list[i] = point_this;

            if (extrinsic_est_en)
            {
                if (!use_imu_as_input)
                {

                    point_this = kf_output.x_.offset_R_L_I.normalized() * point_this + kf_output.x_.offset_T_L_I;
                }
                else
                {

                    point_this = kf_input.x_.offset_R_L_I.normalized() * point_this + kf_input.x_.offset_T_L_I;
                }
            }
            else
            {
                point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
            }

           
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);
            crossmat_list[i]=point_crossmat;
        }

        if (!use_imu_as_input)
        {
            bool imu_upda_cov = false;
            effct_feat_num = 0;

            /**** point by point update ****/

            double pcl_beg_time = Measures.lidar_beg_time;
            idx = -1;
            for (k = 0; k < time_seq.size(); k++)
            {

                PointType &point_body = feats_down_body->points[idx + time_seq[k]];

                time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                begin_bias_diagnostic_frame();
                const LioDiagStateSnapshot ekf_pre_state = snapshot_lio_state(kf_output.x_);

                if (is_first_frame)
                {
                    if (imu_en)
                    {
                        while (time_current > get_time_in_sec(imu_next.header.stamp))
                        {
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                        }

                        angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                    }
                    is_first_frame = false;
                    imu_upda_cov = true;
                    time_update_last = time_current;
                    time_predict_last_const = time_current;
                }

                if (imu_en)
                {
                    bool imu_comes = time_current > get_time_in_sec(imu_next.header.stamp);
                    while (imu_comes)
                    {
                        imu_upda_cov = true;
                        angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                        /*** covariance update ***/
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        double dt = get_time_in_sec(imu_last.header.stamp) - time_predict_last_const;
                        run_bias_diagnostic_operation(
                            kf_output.x_,
                            [&]() { kf_output.predict(dt, Q_output, input_in, true, false); },
                            LIO_BIAS_SOURCE_PROPAGATION);
                        time_predict_last_const = get_time_in_sec(imu_last.header.stamp);
                        imu_comes = time_current > get_time_in_sec(imu_next.header.stamp);

                        {
                            double dt_cov = get_time_in_sec(imu_last.header.stamp) - time_update_last;

                            if (dt_cov > 0.0)
                            {
                                time_update_last = get_time_in_sec(imu_last.header.stamp);
                                double propag_imu_start = omp_get_wtime();

                                run_bias_diagnostic_operation(
                                    kf_output.x_,
                                    [&]() { kf_output.predict(dt_cov, Q_output, input_in, false, true); },
                                    LIO_BIAS_SOURCE_PROPAGATION);

                                propag_time += omp_get_wtime() - propag_imu_start;
                                double solve_imu_start = omp_get_wtime();
                                run_bias_diagnostic_operation(
                                    kf_output.x_,
                                    [&]() { kf_output.update_iterated_dyn_share_IMU(); },
                                    LIO_BIAS_SOURCE_IMU_UPDATE);
                                record_post100_imu_innovation(kf_output);
                                solve_time += omp_get_wtime() - solve_imu_start;
                            }
                        }
                    }
                }

                double dt = time_current - time_predict_last_const;
                double propag_state_start = omp_get_wtime();
                if (!prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        run_bias_diagnostic_operation(
                            kf_output.x_,
                            [&]() { kf_output.predict(dt_cov, Q_output, input_in, false, true); },
                            LIO_BIAS_SOURCE_PROPAGATION);
                        time_update_last = time_current;
                    }
                }
                run_bias_diagnostic_operation(
                    kf_output.x_,
                    [&]() { kf_output.predict(dt, Q_output, input_in, true, false); },
                    LIO_BIAS_SOURCE_PROPAGATION);
                propag_time += omp_get_wtime() - propag_state_start;
                time_predict_last_const = time_current;

                double t_update_start = omp_get_wtime();

                if (feats_down_size < 1)
                {
                    printf("No point, skip this scan!\n");
                    idx += time_seq[k];
                    continue;
                }
                const std::uint64_t nearest_before = lio_diag.nearest_reject;
                const std::uint64_t nearest_too_few_before = lio_diag.nearest_too_few_count;
                const std::uint64_t plane_before = lio_diag.plane_reject;
                const std::uint64_t residual_before = lio_diag.residual_reject;
                lio_diag.ekf_update_attempts++;
                lio_diag_reset_measurement_attempt();
                lio_diag_reset_early_attempt();
                const LioDiagStateSnapshot ekf_propagated_state = snapshot_lio_state(kf_output.x_);
                bool ekf_update_ok = false;
                run_bias_diagnostic_operation(
                    kf_output.x_,
                    [&]() { ekf_update_ok = kf_output.update_iterated_dyn_share_modified(); },
                    LIO_BIAS_SOURCE_LIDAR_UPDATE);
                record_early_ekf_gain(kf_output);
                const LioDiagStateSnapshot ekf_post_state = snapshot_lio_state(kf_output.x_);
                record_ekf_trace(ekf_pre_state, ekf_propagated_state, ekf_post_state, ekf_update_ok);
                report_startup_group(Measures, dt, ekf_update_ok, nearest_before,
                                     nearest_too_few_before, plane_before, residual_before,
                                     ekf_pre_state,
                                     ekf_propagated_state, ekf_post_state);
                if (!ekf_update_ok)
                {
                    finish_bias_diagnostic_frame();
                    lio_diag.ekf_update_fail++;
                    if (!lio_diag.first_ekf_failure_reported)
                    {
                        lio_diag.first_ekf_failure_reported = true;
                        std::cout << "[LIO-DIAG] FIRST_EKF_FAILURE"
                                  << " effect_num=" << lio_diag.effect_num_last
                                  << " nearest_reject=" << lio_diag.nearest_reject
                                  << " plane_reject=" << lio_diag.plane_reject
                                  << " residual_reject=" << lio_diag.residual_reject
                                  << std::endl;
                    }
                    idx = idx + time_seq[k];
                    continue;
                }
                lio_diag.ekf_update_success++;

                if (prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (!imu_en && (dt_cov >= imu_time_inte))
                    {
                        double propag_cov_start = omp_get_wtime();
                        run_bias_diagnostic_operation(
                            kf_output.x_,
                            [&]() { kf_output.predict(dt_cov, Q_output, input_in, false, true); },
                            LIO_BIAS_SOURCE_PROPAGATION);
                        imu_upda_cov = false;
                        time_update_last = time_current;
                        propag_time += omp_get_wtime() - propag_cov_start;
                    }
                }

                finish_bias_diagnostic_frame();

                solve_start = omp_get_wtime();

                if (publish_odometry_without_downsample)
                {
                    /******* Publish odometry *******/

                    publish_odometry(pubOdomAftMapped);
                    if (runtime_pos_log)
                    {
                        state_out = kf_output.x_;
                        euler_cur = SO3ToEuler(state_out.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose()
                                 << " " << state_out.omg.transpose() << " " << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " " << state_out.bg.transpose() << " " << state_out.ba.transpose() << " " << feats_undistort->points.size() << endl;
                    }
                }

                for (int j = 0; j < time_seq[k]; j++)
                {
                    PointType &point_body_j = feats_down_body->points[idx + j + 1];
                    PointType &point_world_j = feats_down_world->points[idx + j + 1];
                    pointBodyToWorld(&point_body_j, &point_world_j);
                }

                solve_time += omp_get_wtime() - solve_start;

                update_time += omp_get_wtime() - t_update_start;
                idx += time_seq[k];
            }
        }
        else
        {
            bool imu_prop_cov = false;
            effct_feat_num = 0;

            double pcl_beg_time = Measures.lidar_beg_time;
            idx = -1;
            for (k = 0; k < time_seq.size(); k++)
            {
                PointType &point_body = feats_down_body->points[idx + time_seq[k]];
                time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                begin_bias_diagnostic_frame();
                const LioDiagStateSnapshot ekf_pre_state = snapshot_lio_state(kf_input.x_);
                if (is_first_frame)
                {
                    while (time_current > get_time_in_sec(imu_next.header.stamp))
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                    }
                    imu_prop_cov = true;

                    is_first_frame = false;
                    t_last = time_current;
                    time_update_last = time_current;

                    {
                        input_in.gyro << imu_last.angular_velocity.x,
                            imu_last.angular_velocity.y,
                            imu_last.angular_velocity.z;

                        input_in.acc << imu_last.linear_acceleration.x,
                            imu_last.linear_acceleration.y,
                            imu_last.linear_acceleration.z;

                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                    }
                }

                while (time_current > get_time_in_sec(imu_next.header.stamp))
                {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    imu_deque.pop_front();
                    input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                    input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                    input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                    double dt = get_time_in_sec(imu_last.header.stamp) - t_last;

                    double dt_cov = get_time_in_sec(imu_last.header.stamp) - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        run_bias_diagnostic_operation(
                            kf_input.x_,
                            [&]() { kf_input.predict(dt_cov, Q_input, input_in, false, true); },
                            LIO_BIAS_SOURCE_PROPAGATION);
                        time_update_last = get_time_in_sec(imu_last.header.stamp);
                    }
                    run_bias_diagnostic_operation(
                        kf_input.x_,
                        [&]() { kf_input.predict(dt, Q_input, input_in, true, false); },
                        LIO_BIAS_SOURCE_PROPAGATION);
                    t_last = get_time_in_sec(imu_last.header.stamp);
                    imu_prop_cov = true;
                }

                double dt = time_current - t_last;
                t_last = time_current;
                double propag_start = omp_get_wtime();

                if (!prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        run_bias_diagnostic_operation(
                            kf_input.x_,
                            [&]() { kf_input.predict(dt_cov, Q_input, input_in, false, true); },
                            LIO_BIAS_SOURCE_PROPAGATION);
                        time_update_last = time_current;
                    }
                }
                run_bias_diagnostic_operation(
                    kf_input.x_,
                    [&]() { kf_input.predict(dt, Q_input, input_in, true, false); },
                    LIO_BIAS_SOURCE_PROPAGATION);

                propag_time += omp_get_wtime() - propag_start;

                double t_update_start = omp_get_wtime();

                if (feats_down_size < 1)
                {
                    printf("No point, skip this scan!\n");

                    idx += time_seq[k];
                    continue;
                }
                const std::uint64_t nearest_before = lio_diag.nearest_reject;
                const std::uint64_t nearest_too_few_before = lio_diag.nearest_too_few_count;
                const std::uint64_t plane_before = lio_diag.plane_reject;
                const std::uint64_t residual_before = lio_diag.residual_reject;
                lio_diag.ekf_update_attempts++;
                lio_diag_reset_measurement_attempt();
                lio_diag_reset_early_attempt();
                const LioDiagStateSnapshot ekf_propagated_state = snapshot_lio_state(kf_input.x_);
                bool ekf_update_ok = false;
                run_bias_diagnostic_operation(
                    kf_input.x_,
                    [&]() { ekf_update_ok = kf_input.update_iterated_dyn_share_modified(); },
                    LIO_BIAS_SOURCE_LIDAR_UPDATE);
                record_early_ekf_gain(kf_input);
                const LioDiagStateSnapshot ekf_post_state = snapshot_lio_state(kf_input.x_);
                record_ekf_trace(ekf_pre_state, ekf_propagated_state, ekf_post_state, ekf_update_ok);
                report_startup_group(Measures, dt, ekf_update_ok, nearest_before,
                                     nearest_too_few_before, plane_before, residual_before,
                                     ekf_pre_state,
                                     ekf_propagated_state, ekf_post_state);
                if (!ekf_update_ok)
                {
                    finish_bias_diagnostic_frame();
                    lio_diag.ekf_update_fail++;
                    if (!lio_diag.first_ekf_failure_reported)
                    {
                        lio_diag.first_ekf_failure_reported = true;
                        std::cout << "[LIO-DIAG] FIRST_EKF_FAILURE"
                                  << " effect_num=" << lio_diag.effect_num_last
                                  << " nearest_reject=" << lio_diag.nearest_reject
                                  << " plane_reject=" << lio_diag.plane_reject
                                  << " residual_reject=" << lio_diag.residual_reject
                                  << std::endl;
                    }
                    idx = idx + time_seq[k];
                    continue;
                }
                lio_diag.ekf_update_success++;

                finish_bias_diagnostic_frame();

                solve_start = omp_get_wtime();

                if (publish_odometry_without_downsample)
                {
                    /******* Publish odometry *******/

                    publish_odometry(pubOdomAftMapped);
                    if (runtime_pos_log)
                    {
                        state_in = kf_input.x_;
                        euler_cur = SO3ToEuler(state_in.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose()
                                 << " " << state_in.bg.transpose() << " " << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " " << feats_undistort->points.size() << endl;
                    }
                }

                for (int j = 0; j < time_seq[k]; j++)
                {
                    PointType &point_body_j = feats_down_body->points[idx + j + 1];
                    PointType &point_world_j = feats_down_world->points[idx + j + 1];
                    pointBodyToWorld(&point_body_j, &point_world_j);
                }
                solve_time += omp_get_wtime() - solve_start;

                update_time += omp_get_wtime() - t_update_start;
                idx = idx + time_seq[k];
            }
        }

        update_lio_diag_scan_bbox();

        /******* Publish odometry downsample *******/
        if (!publish_odometry_without_downsample)
        {
            publish_odometry(pubOdomAftMapped);
        }

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();

        if (feats_down_size > 4)
        {
            map_incremental();
        }

        // Emit exactly one aggregate record for the synchronized LiDAR scan,
        // after all of its internal time slices / EKF calls have completed.
        report_lidar_frame_diagnostic();

        t5 = omp_get_wtime();

        /******* Publish points *******/

        if (path_en)
            publish_path(pubPath);

        if (scan_pub_en || pcd_save_en)
            publish_frame_world(pubLaserCloudFullRes);

        if (scan_pub_en && scan_body_pub_en)
            publish_frame_body(pubLaserCloudFullRes_body);

        /*** Debug variables Logging ***/
        if (runtime_pos_log)
        {
            frame_num++;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            {
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num;
            }
            aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
            aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time / frame_num;
            aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
            T1[time_log_counter] = Measures.lidar_beg_time;
            s_plot[time_log_counter] = t5 - t0;
            s_plot2[time_log_counter] = feats_undistort->points.size();
            s_plot3[time_log_counter] = aver_time_consu;
            time_log_counter++;
            printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_propag);
            if (!publish_odometry_without_downsample)
            {
                if (!use_imu_as_input)
                {
                    state_out = kf_output.x_;
                    euler_cur = SO3ToEuler(state_out.rot);
                    fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose()
                             << " " << state_out.omg.transpose() << " " << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " " << state_out.bg.transpose() << " " << state_out.ba.transpose() << " " << feats_undistort->points.size() << endl;
                }
                else
                {
                    state_in = kf_input.x_;
                    euler_cur = SO3ToEuler(state_in.rot);
                    fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose()
                             << " " << state_in.bg.transpose() << " " << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " " << feats_undistort->points.size() << endl;
                }
            }
            dump_lio_state_to_log(fp);
        }

        rate.sleep();
    }

    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/

    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        std::cout << "Saving map to file: " << all_points_dir << std::endl;
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    fout_out.close();
    fout_imu_pbp.close();

    return 0;
}
