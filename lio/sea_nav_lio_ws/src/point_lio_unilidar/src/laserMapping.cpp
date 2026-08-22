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
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_deque;

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

void report_first_twenty_group(double dt, bool update_ok,
                               std::uint64_t nearest_before,
                               std::uint64_t plane_before,
                               std::uint64_t residual_before)
{
    if (!init_map || lio_diag.post_map_group_count >= 20)
        return;

    Eigen::Vector3d pos, vel, gravity, acc_bias, gyro_bias;
    V3D euler_deg;
    get_lio_diag_state(pos, vel, gravity, acc_bias, gyro_bias, euler_deg);

    const std::uint64_t nearest_delta = lio_diag.nearest_reject - nearest_before;
    const std::uint64_t plane_delta = lio_diag.plane_reject - plane_before;
    const std::uint64_t residual_delta = lio_diag.residual_reject - residual_before;
    const char *reason = "matched_or_other";
    if (nearest_delta > 0)
        reason = "nearest";
    else if (plane_delta > 0)
        reason = "plane";
    else if (residual_delta > 0)
        reason = "residual";
    else if (!update_ok)
        reason = "invalid";

    std::cout << "[LIO-DIAG] FIRST20_GROUP"
              << " index=" << lio_diag.post_map_group_count
              << " dt=" << dt
              << " state_pos=" << pos.transpose()
              << " state_vel=" << vel.transpose()
              << " gravity=" << gravity.transpose()
              << " effect_num=" << lio_diag.effect_num_last
              << " update=" << (update_ok ? "SUCCESS" : "FAIL")
              << " reject_reason=" << reason
              << " nearest_reject=" << nearest_delta
              << " plane_reject=" << plane_delta
              << " residual_reject=" << residual_delta
              << std::endl;
    lio_diag.post_map_group_count++;
}

void report_lio_diagnostics_if_due()
{
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

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = get_time_in_sec(msg->header.stamp);

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());

    double time_div = get_time_in_sec(msg->header.stamp);

    p_pre->process(msg, ptr);

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
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }

        if (!ptr_div->empty())
        {
            lidar_buffer.push_back(ptr_div);

            time_buffer.push_back(time_div);
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
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else
    {
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(get_time_in_sec(msg->header.stamp));
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

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    imu_deque.emplace_back(msg);

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
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_buffer.pop_front();

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

        if (meas.lidar->points.size() < 1)
        {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
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

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = get_time_in_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time)
                break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
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
                kf_input.change_x(state_in);
                kf_output.change_x(state_out);
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
                        kf_output.predict(dt, Q_output, input_in, true, false);
                        time_predict_last_const = get_time_in_sec(imu_last.header.stamp);
                        imu_comes = time_current > get_time_in_sec(imu_next.header.stamp);

                        {
                            double dt_cov = get_time_in_sec(imu_last.header.stamp) - time_update_last;

                            if (dt_cov > 0.0)
                            {
                                time_update_last = get_time_in_sec(imu_last.header.stamp);
                                double propag_imu_start = omp_get_wtime();

                                kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                propag_time += omp_get_wtime() - propag_imu_start;
                                double solve_imu_start = omp_get_wtime();
                                kf_output.update_iterated_dyn_share_IMU();
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
                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        time_update_last = time_current;
                    }
                }
                kf_output.predict(dt, Q_output, input_in, true, false);
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
                const std::uint64_t plane_before = lio_diag.plane_reject;
                const std::uint64_t residual_before = lio_diag.residual_reject;
                lio_diag.ekf_update_attempts++;
                const LioDiagStateSnapshot ekf_propagated_state = snapshot_lio_state(kf_output.x_);
                const bool ekf_update_ok = kf_output.update_iterated_dyn_share_modified();
                const LioDiagStateSnapshot ekf_post_state = snapshot_lio_state(kf_output.x_);
                record_ekf_trace(ekf_pre_state, ekf_propagated_state, ekf_post_state, ekf_update_ok);
                report_first_twenty_group(dt, ekf_update_ok, nearest_before, plane_before, residual_before);
                if (!ekf_update_ok)
                {
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
                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        imu_upda_cov = false;
                        time_update_last = time_current;
                        propag_time += omp_get_wtime() - propag_cov_start;
                    }
                }

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
                        kf_input.predict(dt_cov, Q_input, input_in, false, true);
                        time_update_last = get_time_in_sec(imu_last.header.stamp);
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);
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
                        kf_input.predict(dt_cov, Q_input, input_in, false, true);
                        time_update_last = time_current;
                    }
                }
                kf_input.predict(dt, Q_input, input_in, true, false);

                propag_time += omp_get_wtime() - propag_start;

                double t_update_start = omp_get_wtime();

                if (feats_down_size < 1)
                {
                    printf("No point, skip this scan!\n");

                    idx += time_seq[k];
                    continue;
                }
                const std::uint64_t nearest_before = lio_diag.nearest_reject;
                const std::uint64_t plane_before = lio_diag.plane_reject;
                const std::uint64_t residual_before = lio_diag.residual_reject;
                lio_diag.ekf_update_attempts++;
                const LioDiagStateSnapshot ekf_propagated_state = snapshot_lio_state(kf_input.x_);
                const bool ekf_update_ok = kf_input.update_iterated_dyn_share_modified();
                const LioDiagStateSnapshot ekf_post_state = snapshot_lio_state(kf_input.x_);
                record_ekf_trace(ekf_pre_state, ekf_propagated_state, ekf_post_state, ekf_update_ok);
                report_first_twenty_group(dt, ekf_update_ok, nearest_before, plane_before, residual_before);
                if (!ekf_update_ok)
                {
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
