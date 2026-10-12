#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

// 首层读取原始 PCD 时使用的帧编号最小位宽，由 ROS 私有参数设置。
int pcd_name_fill_num = 0;

// 将一帧点云变换到世界坐标系，并按体素归入共享的八叉树根节点。
// 同一窗口内落入同一体素的多帧点会在一个节点中共同形成几何约束。
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                pcl::PointCloud<PointType>& feat_pt,
                Eigen::Quaterniond q, Eigen::Vector3d t, int fnum,
                double voxel_size, int window_size, float eigen_ratio)
{
  float loc_xyz[3];
  // 对当前帧每个点独立计算世界坐标与体素归属，并累加到对应帧槽位 fnum。
  for(PointType& p_c: feat_pt.points)
  {
    // 保留原始传感器坐标，同时计算世界坐标以确定全局体素位置。
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = q * pvec_orig + t;

    // 当前实现对负坐标减 1 再转整数，避免一般负小数被向零截断到相邻体素。
    // 对 x、y、z 三个坐标分别按 voxel_size 取体素索引。
    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    // 已存在该体素时追加当前帧的点和统计量。
    if(iter != feat_map.end())
    {
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);

      iter->second->sig_orig[fnum].push(pvec_orig);
      iter->second->sig_tran[fnum].push(pvec_tran);
    }
    else
    {
      // 首次遇到该体素时创建根节点，并初始化其空间范围和八叉树层级。
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);

      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

// 普通工作线程处理分配到的完整滑动窗口。窗口含 WIN_SIZE 帧，起点间隔为 GAP。
void parallel_comp(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  // i 是本线程负责的窗口序号；窗口原始帧范围从 i*GAP 延伸 WIN_SIZE 帧。
  for(int i = thread_id*part_length; i < (thread_id+1)*part_length; i++)
  {
    // 首层从磁盘读取原始帧；更高层使用前一层生成的关键帧点云。
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE); raw_pc.resize(WIN_SIZE);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    // 将当前窗口的初始位姿转换为优化器使用的旋转矩阵和平移向量。
    // j 是窗口局部帧号；i*GAP+j 是该帧在本层完整轨迹中的全局索引。
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    // 高层的 pcds 按整层关键帧编号存储，因此这里以全局 j 取点云、以 j-i*GAP 存入局部数组。
    if(layer_num != 1)
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      // 原始点云只在第一次迭代加载；每轮由 raw_pc 重新复制可避免降采样修改原始数据。
      // 外层 loop 重建体素和优化位姿；首层内层 j 遍历当前窗口对应的原始 PCD 帧号。
      if(layer_num == 1)
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();
        }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
      // 按当前位姿将各帧点云降采样并汇总到共同的体素/八叉树结构。
      // j 同时是局部帧槽位，供 cut_voxel 把点统计量放入该帧对应的向量元素。
      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
      }
      // 对所有根体素递归判平面或细分八叉树。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      
      // 将平面体素转换为 Hessian 约束，剔除高残差约束后执行阻尼优化。
      VOX_HESS voxhess(WIN_SIZE);
      // 每个根节点把其平面叶节点注册到当前窗口的共享约束集合。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      // 优化结束后逐个释放本轮创建的八叉树根节点及其子树。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      
      // 残差相对变化小于 5% 时提前收敛；否则在达到 max_iter 时结束。
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;
        
        // 窗口内每一对帧对应一个 6 维 Hessian 摘要，连续写入该窗口的存储区间。
        for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        
        break;
      }
      residual_pre = residual_cur;
    }

    // 将窗口内各帧变换到窗口首帧坐标系，合并和降采样后作为下一层的一帧。
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    // 将局部数组中的每帧点云用优化后相对位姿对齐到第 0 帧坐标系，再追加到输出云。
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
}

// 最后一个工作线程处理分配到的完整窗口及不足一个完整窗口的尾部帧。
void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  int& left_gap_num = layer.left_gap_num;

  double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0,
    tran_t = 0, sol_t = 0, save_t = 0;
  
  // 检查最后线程负责的完整窗口数量是否与分层参数计算一致。
  if(layer.gap_num-(layer.thread_num-1)*part_length+1!=left_gap_num) printf("THIS IS WRONG!\n");

  // 先处理完整窗口；每个阶段分别累计耗时用于性能统计。
  // i 从最后线程分配区间起点开始，最多处理 left_gap_num 个完整窗口。
  for(uint i = thread_id*part_length; i < thread_id*part_length+left_gap_num; i++)
  {
    printf("parallel computing %d\n", i);
    double t0, t1;
    double t_begin = ros::Time::now().toSec();
    
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE); raw_pc.resize(WIN_SIZE);
    
    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    // 将全局轨迹中本窗口的 WIN_SIZE 个位姿复制到局部优化状态数组。
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    if(layer_num != 1)
    {
      t0 = ros::Time::now().toSec();
      // 从整层点云数组截取当前窗口帧，局部索引仍为 j-i*GAP。
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
      load_t += ros::Time::now().toSec()-t0;
    }

    size_t mem_cost = 0;
    // 每轮用当前 x_buf 重新生成体素约束并更新窗口位姿，直到收敛或达到迭代上限。
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      // 首层载入原始帧，其余层复用前一层生成的点云。
      if(layer_num == 1)
      {
        t0 = ros::Time::now().toSec();
        // j 遍历该窗口的全局 PCD 编号；首次迭代读盘，所有迭代都从原始缓存复制。
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();
        }
        load_t += ros::Time::now().toSec()-t0;
      }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      // 构建窗口的体素几何约束，并累计降采样、切分和重切分耗时。
      // 每个 j 处理一个窗口局部帧，分别计量降采样和体素归属时间。
      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        t0 = ros::Time::now().toSec();
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        dsp_t += ros::Time::now().toSec()-t0;

        t0 = ros::Time::now().toSec();
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
        cut_t += ros::Time::now().toSec()-t0;
      }

      t0 = ros::Time::now().toSec();
      // 遍历本窗口所有根体素，递归确定有效平面叶节点。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      recut_t += ros::Time::now().toSec()-t0;

      t0 = ros::Time::now().toSec();
      VOX_HESS voxhess(WIN_SIZE);
      // 收集每个根节点中的平面约束并记录转换耗时。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);
      tran_t += ros::Time::now().toSec()-t0;

      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      t0 = ros::Time::now().toSec();
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      sol_t += ros::Time::now().toSec()-t0;

      // 本轮约束已复制/引用到优化阶段使用完毕，释放整棵体素树。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;

      // 达到残差收敛条件或迭代上限后，保存当前窗口的 Hessian 约束。
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        if(i < thread_id*part_length+left_gap_num)
          // 按当前窗口编号把帧对 Hessian 摘要写入该窗口专属区间。
          for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
            layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }
    
    // 对齐并合并窗口点云，生成下一层关键帧，同时记录输出阶段耗时。
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    // 逐帧变换到窗口首帧坐标系；每次循环计量单帧变换及追加耗时。
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      t1 = ros::Time::now().toSec();
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
      save_t += ros::Time::now().toSec()-t1;
    }
    t0 = ros::Time::now().toSec();
    downsample_voxel(*pc_keyframe, 0.05);
    dsp_t += ros::Time::now().toSec()-t0;

    t0 = ros::Time::now().toSec();
    next_layer.pcds[i] = pc_keyframe;
    save_t += ros::Time::now().toSec()-t0;
    
    total_t += ros::Time::now().toSec()-t_begin;
  }

  // pose 数无法组成完整窗口时，用实际剩余帧数 last_win_size 单独优化尾窗。
  if(layer.tail > 0)
  {
    int i = thread_id*part_length+left_gap_num;

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(layer.last_win_size); raw_pc.resize(layer.last_win_size);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(layer.last_win_size);
    // 尾窗不满 WIN_SIZE，只复制 last_win_size 个有效位姿。
    for(int j = 0; j < layer.last_win_size; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }

    if(layer_num != 1)
    {
      // 从高层点云数组按尾窗全局帧号取出实际存在的点云。
      for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
    }

    size_t mem_cost = 0;
    // 短尾窗同样按当前位姿反复重建约束，但优化器维度只取 last_win_size 帧。
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      // 尾窗的点云来源与完整窗口相同，但读取和优化的帧数使用 last_win_size。
      if(layer_num == 1)
        // 读取尾窗覆盖的末尾 PCD 帧；局部数组下标从 0 到 last_win_size-1。
        for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();          
        }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      // 使用尾窗实际帧数建立体素约束和优化器。
      // 只处理尾窗实际包含的帧，不访问完整窗口之外的无效槽位。
      for(size_t j = 0; j < layer.last_win_size; j++)
      {
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
      }
      // 遍历尾窗的根体素，递归构建其平面叶节点。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      
      VOX_HESS voxhess(layer.last_win_size);
      // 将尾窗的平面叶节点收集为优化器约束。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(layer.last_win_size);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      // 释放尾窗本轮创建的所有根节点及其子树。
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      
      // 保存尾窗的 Hessian；数组步长保持与完整窗口的存储布局一致。
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        // 尾窗只保存其实际帧对的摘要，目标地址仍按完整窗口容量计算偏移。
        for(int j = 0; j < layer.last_win_size*(layer.last_win_size-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        
        break;
      }
      residual_pre = residual_cur;
    }

    // 尾窗输出也变换到其首帧坐标系，合并降采样后交给下一层。
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    // 遍历尾窗真实帧数并把各帧点云对齐、追加到尾窗关键帧点云。
    for(size_t j = 0; j < layer.last_win_size; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
  // 输出最后线程完整窗口处理阶段的累计耗时和占比；短尾窗不计入这些计时器。
  printf("total time: %.2fs\n", total_t);
  printf("load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
   "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% | "
   "save pcd %.2fs %.2f%%\n",
    load_t, load_t/total_t*100, undis_t, undis_t/total_t*100,
    dsp_t, dsp_t/total_t*100, cut_t, cut_t/total_t*100, recut_t, recut_t/total_t*100, tran_t, tran_t/total_t*100,
    sol_t, sol_t/total_t*100, save_t, save_t/total_t*100);
}

// 在最顶层对所有位姿和点云联合执行全局 BA。
void global_ba(LAYER& layer)
{
  // 复制当前层状态到优化工作区，迭代期间更新 x_buf，结束后再写回 pose_vec。
  int window_size = layer.pose_vec.size();
  vector<IMUST> x_buf(window_size);
  // 把整层四元数和平移复制为全局优化器使用的矩阵/向量状态。
  for(int i = 0; i < window_size; i++)
  {
    x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[i].t;
  }

  vector<pcl::PointCloud<PointType>::Ptr> src_pc;
  src_pc.resize(window_size);
  // 为全局优化准备每个轨迹位姿对应的一帧点云副本。
  for(int i = 0; i < window_size; i++)
    src_pc[i] = (*layer.pcds[i]).makeShared();

  double residual_cur = 0, residual_pre = 0;
  size_t mem_cost = 0, max_mem = 0;
  double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0, t0;
  // 每轮基于最新全局位姿重建所有体素约束，再执行一次全局状态更新。
  for(int loop = 0; loop < layer.max_iter; loop++)
  {
    std::cout<<"---------------------"<<std::endl;
    std::cout<<"Iteration "<<loop<<std::endl;

    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    // 全部帧共同建立体素约束，不再按局部滑动窗口拆分。
    // 逐帧将点变换并累加到共享体素表；此处窗口大小等于整层位姿数。
    for(int i = 0; i < window_size; i++)
    {
      t0 = ros::Time::now().toSec();
      if(layer.downsample_size > 0) downsample_voxel(*src_pc[i], layer.downsample_size);
      dsp_t += ros::Time::now().toSec() - t0;
      t0 = ros::Time::now().toSec();
      cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i,
                layer.voxel_size, window_size, layer.eigen_ratio*2);
      cut_t += ros::Time::now().toSec() - t0;
    }
    t0 = ros::Time::now().toSec();
    // 对全局体素逐个执行八叉树平面判定和必要的递归切分。
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();
    recut_t += ros::Time::now().toSec() - t0;
    
    t0 = ros::Time::now().toSec();
    VOX_HESS voxhess(window_size);
    // 把所有根节点中的平面叶节点转换为全局优化约束。
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->tras_opt(voxhess);
    tran_t += ros::Time::now().toSec() - t0;
    
    t0 = ros::Time::now().toSec();
    VOX_OPTIMIZER opt_lsv(window_size);
    opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6) hess_vec;
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
    sol_t += ros::Time::now().toSec() - t0;

    // 全局优化本轮完成后释放所有体素树，下一轮会按更新位姿重新构造。
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;
    
    // 输出残差变化；满足收敛条件或达到迭代上限时保存 Hessian 信息。
    cout<<"Residual absolute: "<<abs(residual_pre-residual_cur)<<" | "
      <<"percentage: "<<abs(residual_pre-residual_cur)/abs(residual_cur)<<endl;
    
    if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
    {
      if(max_mem < mem_cost) max_mem = mem_cost;
      #ifdef FULL_HESS
      // 每个全局位姿对都有一个 6 维 Hessian 摘要，依次写入顶层约束数组。
      for(int i = 0; i < window_size*(window_size-1)/2; i++)
        layer.hessians[i] = hess_vec[i];
      #else
      // 未启用 FULL_HESS 时只逐帧导出相邻帧间的 6x6 Hessian 块。
      for(int i = 0; i < window_size-1; i++)
      {
        Matrix6d hess = Hess_cur.block(6*i, 6*i+6, 6, 6);
        // 将该块的 36 个元素按行写入文件，以空格分隔。
        for(int row = 0; row < 6; row++)
          for(int col = 0; col < 6; col++)
            hessFile << hess(row, col) << ((row*col==25)?"":" ");
        if(i < window_size-2) hessFile << "\n";
      }
      #endif
      break;
    }
    residual_pre = residual_cur;
  }
  // 将全局优化得到的位姿写回该层。
  // x_buf 与 pose_vec 使用同一帧索引，逐帧写回旋转和平移。
  for(int i = 0; i < window_size; i++)
  {
    layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
    layer.pose_vec[i].t = x_buf[i].p;
  }
  printf("Downsample: %f, Cut: %f, Recut: %f, Tras: %f, Sol: %f\n", dsp_t, cut_t, recut_t, tran_t, sol_t);
}

// 启动分层处理线程：前面的线程处理固定数量窗口，最后一个线程接管剩余窗口和尾窗。
void distribute_thread(LAYER& layer, LAYER& next_layer)
{
  int& thread_num = layer.thread_num;
  double t0 = ros::Time::now().toSec();
  // 为每个线程创建任务；普通线程处理完整分块，最后一个线程走余量处理函数。
  for(int i = 0; i < thread_num; i++)
    if(i < thread_num-1)
      layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer));
    else
      layer.mthreads[i] = new thread(parallel_tail, ref(layer), i, ref(next_layer));
  // printf("Thread distribution time: %f\n", ros::Time::now().toSec()-t0);

  t0 = ros::Time::now().toSec();
  // 等待所有工作线程完成并释放线程对象，之后才能安全更新下一层。
  // join 保证线程写入当前层/下一层的数据在函数返回前全部完成。
  for(int i = 0; i < thread_num; i++)
  {
    layer.mthreads[i]->join();
    delete layer.mthreads[i];
  }
  // printf("Thread join time: %f\n", ros::Time::now().toSec()-t0);
}

// ROS 节点入口：读取配置，逐层运行局部 BA，最后运行全局 BA 和位姿图优化。
int main(int argc, char** argv)
{
	ros::init(argc, argv, "hba");
	ros::NodeHandle nh("~");

  int total_layer_num, thread_num;
  string data_path;

  // 读取层数、PCD 编号格式、数据目录和线程数等 ROS 私有参数。
  nh.getParam("total_layer_num", total_layer_num);
  // PCD 帧编号的最小位宽；大于 0 时左侧补零（如 5 -> 00001.pcd），否则使用未补零编号（1.pcd）。
  nh.getParam("pcd_name_fill_num", pcd_name_fill_num);
  nh.getParam("data_path", data_path);
  nh.getParam("thread_num", thread_num);

  HBA hba(total_layer_num, data_path, thread_num);
  // 每层完成后将关键帧状态传递到下一层，作为更粗层级的初始估计。
  // 从第一层运行到倒数第二层；每轮同步等待该层窗口优化，再更新下一层状态。
  for(int i = 0; i < total_layer_num-1; i++)
  {
    std::cout<<"---------------------"<<std::endl;
    distribute_thread(hba.layers[i], hba.layers[i+1]);
    hba.update_next_layer_state(i);
  }
  // 顶层全局 BA 生成约束后，在原始轨迹上运行位姿图优化并写出结果。
  global_ba(hba.layers[total_layer_num-1]);
  hba.pose_graph_optimization();
  printf("iteration complete\n");
}