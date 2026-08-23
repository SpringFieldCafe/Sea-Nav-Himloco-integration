#ifndef Estimator_H
#define Estimator_H

#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

#include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "common_lib.h"
#include "parameters.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <ikd-Tree/ikd_Tree.h>
#include <pcl/io/pcd_io.h>

extern PointCloudXYZI::Ptr normvec; //(new PointCloudXYZI(100000, 1));
extern std::vector<int> time_seq;
extern PointCloudXYZI::Ptr feats_down_body; 
extern PointCloudXYZI::Ptr feats_down_world; 
extern std::vector<V3D> pbody_list;
extern std::vector<PointVector> Nearest_Points; 
extern KD_TREE<PointType> ikdtree;
extern std::vector<float> pointSearchSqDis;
extern bool point_selected_surf[100000]; // = {0};
extern std::vector<M3D> crossmat_list;
extern int effct_feat_num;
extern int k;
extern int idx;
extern V3D angvel_avr, acc_avr;

enum LioGeometryAxis
{
    LIO_GEOMETRY_AXIS_X = 0,
    LIO_GEOMETRY_AXIS_Y = 1,
    LIO_GEOMETRY_AXIS_Z = 2,
};

enum LioGeometryOutcome
{
    LIO_GEOMETRY_NEAREST_REJECT = 0,
    LIO_GEOMETRY_PLANE_REJECT = 1,
    LIO_GEOMETRY_RESIDUAL_REJECT = 2,
    LIO_GEOMETRY_ACCEPTED = 3,
    LIO_GEOMETRY_OTHER_REJECT = 4,
};

struct LioDiagnosticCounters
{
    std::uint64_t frame_count = 0;
    std::uint64_t sync_package_ok = 0;
    std::uint64_t sync_package_fail = 0;
    std::uint64_t imu_process_count = 0;
    std::uint64_t undistort_samples = 0;
    std::uint64_t undistort_sum = 0;
    std::uint64_t down_samples = 0;
    std::uint64_t down_sum = 0;
    std::uint64_t time_groups_last = 0;
    std::uint64_t odom_publish_count = 0;
    std::uint64_t tf_publish_count = 0;
    std::uint64_t registered_cloud_publish_count = 0;

    // Read-only startup input diagnostics. These fields only observe the first
    // ten seconds of received cloud/IMU data and never feed the estimator.
    bool startup_input_started = false;
    bool startup_input_window_complete = false;
    bool startup_input_reported = false;
    double startup_first_cloud_time = 0.0;
    double startup_last_cloud_time = 0.0;
    double startup_map_initialized_time = 0.0;
    std::uint64_t startup_initial_map_points = 0;
    std::uint64_t startup_cloud_count = 0;
    std::uint64_t startup_imu_count = 0;
    std::uint64_t startup_sync_success_count = 0;
    std::uint64_t startup_cloud_point_sum = 0;
    std::uint64_t startup_cloud_point_min = 0;
    std::uint64_t startup_cloud_point_max = 0;
    Eigen::Vector3d startup_cloud_xyz_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d startup_cloud_xyz_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Vector3d startup_cloud_xyz_sum = Eigen::Vector3d::Zero();
    std::uint64_t startup_cloud_xyz_count = 0;
    Eigen::Vector3d startup_imu_acc_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d startup_imu_acc_sq_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d startup_imu_gyro_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d startup_imu_gyro_sq_sum = Eigen::Vector3d::Zero();
    std::vector<double> startup_cloud_timestamps;
    std::vector<double> startup_imu_timestamps;
    std::vector<double> startup_cloud_imu_offsets;
    bool map_initialized_reported = false;
    std::uint64_t nearest_reject = 0;
    std::uint64_t nearest_query_count = 0;
    std::uint64_t nearest_too_few_count = 0;
    std::uint64_t nearest_too_far_count = 0;
    std::uint64_t nearest_other_reject_count = 0;
    std::vector<double> nearest_success_distances;
    std::uint64_t plane_reject = 0;
    std::uint64_t residual_reject = 0;
    std::uint64_t effect_num_samples = 0;
    std::uint64_t effect_num_sum = 0;
    std::uint64_t effect_num_zero_count = 0;
    Eigen::Matrix3d plane_normal_outer_sum = Eigen::Matrix3d::Zero();
    std::uint64_t plane_normal_count = 0;
    std::uint64_t normal_z_dominant_count = 0;
    std::uint64_t prematch_normal_count = 0;
    std::uint64_t prematch_normal_axis_count[3] = {0, 0, 0};
    std::uint64_t geometry_candidate_count[3] = {0, 0, 0};
    std::uint64_t geometry_outcome_count[3][5] = {{0, 0, 0, 0, 0},
                                                   {0, 0, 0, 0, 0},
                                                   {0, 0, 0, 0, 0}};
    std::uint64_t geometry_unlabeled_candidates = 0;
    std::uint64_t geometry_unlabeled_outcome_count[5] = {0, 0, 0, 0, 0};
    std::uint64_t early_geometry_candidate_count[3] = {0, 0, 0};
    std::uint64_t early_geometry_outcome_count[3][5] = {{0, 0, 0, 0, 0},
                                                         {0, 0, 0, 0, 0},
                                                         {0, 0, 0, 0, 0}};
    std::uint64_t early_geometry_unlabeled_candidates = 0;
    std::uint64_t early_geometry_unlabeled_outcome_count[5] = {0, 0, 0, 0, 0};
    std::vector<double> early_abs_residuals[3];
    bool early_geometry_attempt_seen = false;
    std::uint64_t normal_x_dominant_count = 0;
    std::uint64_t normal_y_dominant_count = 0;
    std::uint64_t ekf_update_attempts = 0;
    std::uint64_t ekf_update_success = 0;
    std::uint64_t ekf_update_fail = 0;
    std::uint64_t early_ekf_gain_attempts = 0;
    std::uint64_t early_ekf_gain_samples = 0;
    std::vector<double> early_ekf_abs_residuals;
    std::vector<double> early_ekf_rot_delta_deg;
    std::vector<double> early_ekf_pos_delta_m;
    std::vector<double> early_ekf_k_rot_norm;
    std::vector<double> early_ekf_effective_r;
    Eigen::Vector3d early_ekf_p_rot_diag_start = Eigen::Vector3d::Zero();
    Eigen::Vector3d early_ekf_p_rot_diag_end = Eigen::Vector3d::Zero();
    Eigen::Vector3d early_ekf_p_pos_diag_start = Eigen::Vector3d::Zero();
    Eigen::Vector3d early_ekf_p_pos_diag_end = Eigen::Vector3d::Zero();
    bool early_ekf_cov_start_valid = false;
    bool early_ekf_gain_reported = false;
    bool first_ekf_failure_reported = false;
    std::uint64_t post_map_group_count = 0;
    int effect_num_last = 0;

    // First thirty post-map synchronizer groups.
    std::uint64_t startup_group_count = 0;
    std::uint64_t startup_first_success_group = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t startup_groups_with_zero_imu = 0;
    std::uint64_t startup_groups_with_large_imu_cloud_gap = 0;
    double startup_max_imu_cloud_gap_s = 0.0;
    bool startup_group_summary_reported = false;

    struct RollingMeasurementGroup
    {
        std::uint64_t group_id = 0;
        double lidar_begin_time = 0.0;
        double lidar_end_time = 0.0;
        std::uint64_t cloud_point_count = 0;
        std::uint64_t ekf_attempts = 0;
        std::uint64_t ekf_success = 0;
        int effect_num = 0;
        std::uint64_t nearest_too_few = 0;
        std::uint64_t plane_reject = 0;
        std::uint64_t residual_reject = 0;
        std::uint64_t accepted_point_count = 0;
        double accepted_signed_residual_mean = 0.0;
        double accepted_abs_residual_mean = 0.0;
        double accepted_abs_residual_p50 = 0.0;
        double accepted_abs_residual_p95 = 0.0;
        double accepted_abs_residual_max = 0.0;
        V3D normal_mean = V3D::Zero();
        V3D normal_abs_mean = V3D::Zero();
        std::uint64_t normal_x_dominant_count = 0;
        std::uint64_t normal_y_dominant_count = 0;
        std::uint64_t normal_z_dominant_count = 0;
        double delta_position_norm = 0.0;
        double delta_rotation_deg = 0.0;
        double delta_velocity_norm = 0.0;
        double delta_gyro_bias_norm = 0.0;
        double delta_acc_bias_norm = 0.0;
        V3D pre_position = V3D::Zero();
        V3D post_position = V3D::Zero();
        V3D pre_velocity = V3D::Zero();
        V3D post_velocity = V3D::Zero();
    };

    std::deque<RollingMeasurementGroup> rolling_groups;
    bool drift_0p2_reported = false;
    bool stable_exit_context_reported = false;

    // One record per synchronized LiDAR scan. The older startup_group_count
    // remains an EKF/time-slice counter for backwards-compatible logs.
    std::uint64_t lidar_frame_id = 0;
    bool frame_diag_active = false;
    double frame_diag_lidar_begin_time = 0.0;
    double frame_diag_lidar_end_time = 0.0;
    std::uint64_t frame_diag_point_count = 0;
    std::uint64_t frame_diag_ekf_call_count = 0;
    bool frame_diag_ekf_success_any = false;
    std::uint64_t frame_diag_total_effect_num = 0;
    std::uint64_t frame_diag_total_accepted_points = 0;
    std::uint64_t frame_diag_total_nearest_reject = 0;
    std::uint64_t frame_diag_total_plane_reject = 0;
    std::uint64_t frame_diag_total_residual_reject = 0;
    Eigen::Matrix<double, 12, 12> frame_diag_hth_sum = Eigen::Matrix<double, 12, 12>::Zero();
    double frame_diag_signed_residual_sum = 0.0;
    double frame_diag_signed_residual_sq_sum = 0.0;
    std::vector<double> frame_diag_abs_residuals;
    bool frame_diag_state_valid = false;
    V3D frame_diag_pre_position = V3D::Zero();
    V3D frame_diag_post_position = V3D::Zero();
    V3D frame_diag_pre_velocity = V3D::Zero();
    V3D frame_diag_post_velocity = V3D::Zero();
    double frame_diag_delta_rotation_deg = 0.0;

    // Accepted point-to-plane measurements for the current EKF update. These
    // are diagnostic-only and are reset immediately before each update.
    std::uint64_t current_accepted_point_count = 0;
    double current_signed_residual_sum = 0.0;
    std::vector<double> current_signed_residuals;
    double current_abs_residual_sum = 0.0;
    double current_abs_residual_max = 0.0;
    std::vector<double> current_abs_residuals;
    V3D current_normal_sum = V3D::Zero();
    V3D current_normal_abs_sum = V3D::Zero();
    std::uint64_t current_normal_x_dominant_count = 0;
    std::uint64_t current_normal_y_dominant_count = 0;
    std::uint64_t current_normal_z_dominant_count = 0;

    // Bias-source accounting is diagnostic-only. It is enabled for the first
    // 100 post-map EKF frames and does not alter any filter state or equation.
    bool bias_diag_init_valid = false;
    bool bias_diag_frame_active = false;
    bool bias_diag_reported = false;
    std::uint64_t bias_diag_frame_count = 0;
    V3D bias_diag_frame_start_gyro = V3D::Zero();
    V3D bias_diag_frame_start_acc = V3D::Zero();
    V3D bias_diag_frame_prop_gyro = V3D::Zero();
    V3D bias_diag_frame_prop_acc = V3D::Zero();
    V3D bias_diag_frame_imu_gyro = V3D::Zero();
    V3D bias_diag_frame_imu_acc = V3D::Zero();
    V3D bias_diag_frame_lidar_gyro = V3D::Zero();
    V3D bias_diag_frame_lidar_acc = V3D::Zero();
    V3D bias_diag_frame_other_gyro = V3D::Zero();
    V3D bias_diag_frame_other_acc = V3D::Zero();
    V3D bias_diag_total_gyro = V3D::Zero();
    V3D bias_diag_total_acc = V3D::Zero();
    V3D bias_diag_end_gyro = V3D::Zero();
    V3D bias_diag_end_acc = V3D::Zero();
    V3D bias_diag_init_reset_gyro = V3D::Zero();
    V3D bias_diag_init_reset_acc = V3D::Zero();
    V3D bias_diag_unaccounted_gyro = V3D::Zero();
    V3D bias_diag_unaccounted_acc = V3D::Zero();
    V3D bias_diag_init_gyro = V3D::Zero();
    V3D bias_diag_init_acc = V3D::Zero();
    V3D bias_diag_origin_gyro = V3D::Zero();
    V3D bias_diag_origin_acc = V3D::Zero();
    bool bias_diag_origin_valid = false;
    V3D bias_diag_from_prop_gyro = V3D::Zero();
    V3D bias_diag_from_prop_acc = V3D::Zero();
    V3D bias_diag_from_imu_gyro = V3D::Zero();
    V3D bias_diag_from_imu_acc = V3D::Zero();
    V3D bias_diag_from_lidar_gyro = V3D::Zero();
    V3D bias_diag_from_lidar_acc = V3D::Zero();
    V3D bias_diag_from_other_gyro = V3D::Zero();
    V3D bias_diag_from_other_acc = V3D::Zero();
    double bias_diag_max_imu_gyro_norm = 0.0;
    double bias_diag_max_lidar_gyro_norm = 0.0;
    double bias_diag_max_imu_acc_norm = 0.0;
    double bias_diag_max_lidar_acc_norm = 0.0;

    bool bias_diag_post100_active = false;
    std::uint64_t bias_diag_post100_frame_count = 0;
    V3D bias_diag_post100_baseline_gyro = V3D::Zero();
    V3D bias_diag_post100_baseline_acc = V3D::Zero();
    V3D bias_diag_post100_frame_start_gyro = V3D::Zero();
    V3D bias_diag_post100_frame_start_acc = V3D::Zero();
    V3D bias_diag_post100_frame_imu_gyro = V3D::Zero();
    V3D bias_diag_post100_frame_imu_acc = V3D::Zero();
    V3D bias_diag_post100_frame_lidar_gyro = V3D::Zero();
    V3D bias_diag_post100_frame_lidar_acc = V3D::Zero();
    V3D bias_diag_post100_frame_prop_gyro = V3D::Zero();
    V3D bias_diag_post100_frame_prop_acc = V3D::Zero();
    V3D bias_diag_post100_frame_other_gyro = V3D::Zero();
    V3D bias_diag_post100_frame_other_acc = V3D::Zero();
    V3D bias_diag_post100_imu_gyro = V3D::Zero();
    V3D bias_diag_post100_imu_acc = V3D::Zero();
    V3D bias_diag_post100_lidar_gyro = V3D::Zero();
    V3D bias_diag_post100_lidar_acc = V3D::Zero();
    V3D bias_diag_post100_prop_gyro = V3D::Zero();
    V3D bias_diag_post100_prop_acc = V3D::Zero();
    V3D bias_diag_post100_other_gyro = V3D::Zero();
    V3D bias_diag_post100_other_acc = V3D::Zero();
    V3D bias_diag_post100_unaccounted_gyro = V3D::Zero();
    V3D bias_diag_post100_unaccounted_acc = V3D::Zero();
    Eigen::Vector3d bias_diag_map_init_pos = Eigen::Vector3d::Zero();
    std::vector<double> bias_diag_post100_imu_acc_innovations;
    std::vector<double> bias_diag_post100_imu_gyro_innovations;
    double bias_diag_post100_max_imu_acc_delta = 0.0;
    double bias_diag_post100_max_imu_gyro_delta = 0.0;

    struct BiasDriftSnapshot
    {
        bool valid = false;
        bool reported = false;
        std::uint64_t frame = 0;
        double position_drift = 0.0;
        double velocity_norm = 0.0;
        V3D acc_bias = V3D::Zero();
        V3D gyro_bias = V3D::Zero();
        V3D acc_delta_from_frame100 = V3D::Zero();
        V3D acc_imu_delta_from_frame100 = V3D::Zero();
        V3D acc_lidar_delta_from_frame100 = V3D::Zero();
        V3D gyro_delta_from_frame100 = V3D::Zero();
        V3D gyro_imu_delta_from_frame100 = V3D::Zero();
        V3D gyro_lidar_delta_from_frame100 = V3D::Zero();
        V3D acc_unaccounted = V3D::Zero();
        V3D gyro_unaccounted = V3D::Zero();
        double imu_acc_innovation_p50 = 0.0;
        double imu_acc_innovation_p95 = 0.0;
        double imu_gyro_innovation_p50 = 0.0;
        double imu_gyro_innovation_p95 = 0.0;
        double max_imu_acc_bias_delta_per_update = 0.0;
        double max_imu_gyro_bias_delta_per_update = 0.0;
    };
    BiasDriftSnapshot bias_diag_drift_snapshots[3];
};

extern LioDiagnosticCounters lio_diag;
extern std::vector<int> lio_diag_prematch_axes;

void lio_diag_record_effect_num(int effect_num);
void lio_diag_record_plane_normal(const V3D &normal);
void lio_diag_record_prematch_normal(const V3D &normal);
int lio_diag_dominant_normal_axis(const V3D &normal);
void lio_diag_record_retention_candidate(std::size_t point_index);
void lio_diag_record_retention_outcome(std::size_t point_index, int outcome);
void lio_diag_record_early_candidate(std::size_t point_index, bool enabled);
void lio_diag_record_early_outcome(std::size_t point_index, int outcome, bool enabled);
void lio_diag_record_early_residual(std::size_t point_index, double absolute_residual, bool enabled);
void lio_diag_reset_early_attempt();
void lio_diag_reset_measurement_attempt();
void lio_diag_record_accepted_measurement(double signed_residual, const V3D &normal);
bool lio_diag_begin_early_attempt();

extern V3D Lidar_T_wrt_IMU; //(Zero3d);
extern M3D Lidar_R_wrt_IMU; //(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_input,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(process_noise_output,
((vect3, vel))
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

extern esekfom::esekf<state_input, 24, input_ikfom> kf_input;
extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;
extern state_input state_in;
extern state_output state_out;
extern input_ikfom input_in;

Eigen::Matrix<double, 24, 24> process_noise_cov_input();

Eigen::Matrix<double, 30, 30> process_noise_cov_output();

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in);

// Eigen::Matrix<double, 24, 12> df_dw_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s);

vect3 SO3ToEuler(const SO3 &orient);

void h_model_input(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void pointBodyToWorld(PointType const * const pi, PointType * const po);

const bool time_list(PointType &x, PointType &y); // {return (x.curvature < y.curvature);};

#endif
