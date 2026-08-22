#ifndef Estimator_H
#define Estimator_H

#include <cstdint>
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
    std::uint64_t normal_x_dominant_count = 0;
    std::uint64_t normal_y_dominant_count = 0;
    std::uint64_t ekf_update_attempts = 0;
    std::uint64_t ekf_update_success = 0;
    std::uint64_t ekf_update_fail = 0;
    bool first_ekf_failure_reported = false;
    std::uint64_t post_map_group_count = 0;
    int effect_num_last = 0;
};

extern LioDiagnosticCounters lio_diag;

void lio_diag_record_effect_num(int effect_num);
void lio_diag_record_plane_normal(const V3D &normal);

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
