#ifndef BA_HPP
#define BA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "tools.hpp"

// 局部 BA 的窗口帧数与滑动步长；相邻窗口因此共享一部分帧。
#define WIN_SIZE 10
#define GAP 5
// AVG_THR 使用平均体素残差；FULL_HESS 保存完整层间 Hessian 约束。
#define AVG_THR
#define FULL_HESS
// #define ENABLE_RVIZ
// #define ENABLE_FILTER

const double one_three = (1.0 / 3.0);

int layer_limit = 2;
int MIN_PT = 15;
int thd_num = 16;

// 汇集一个窗口内所有平面体素的统计量，并负责残差及其 Hessian/Jacobian 计算。
class VOX_HESS
{
public:
  // 每个元素指向一个平面体素的逐帧统计量；origin_points 仅供可选调试/过滤使用。
  vector<const vector<VOX_FACTOR>*> plvec_voxels;
  vector<PLV(3)> origin_points;
  int win_size;

  VOX_HESS(int _win_size = WIN_SIZE): win_size(_win_size){origin_points.resize(win_size);}

  ~VOX_HESS()
  {
    vector<const vector<VOX_FACTOR>*>().swap(plvec_voxels);
  }

  void get_center(const PLV(3)& vec_orig, PLV(3)& origin_points_)
  {
    // 将体素内点坐标复制到调试缓存。
    size_t pt_size = vec_orig.size();
    for(size_t i = 0; i < pt_size; i++)
      origin_points_.emplace_back(vec_orig[i]);
    return;
  }

  void push_voxel(const vector<VOX_FACTOR>* sig_orig, const vector<PLV(3)>* vec_orig)
  {
    // 仅接收至少由两帧共同观测的体素，才能形成帧间优化约束。
    int process_size = 0;
    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        process_size++;

    #ifdef ENABLE_FILTER
    if(process_size < 1) return;

    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        get_center((*vec_orig)[i], origin_points[i]);
    #endif
    
    if(process_size < 2) return;
    
    plvec_voxels.push_back(sig_orig);
  }

  Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
  {
    // 计算特征向量外积对称化后的 6 个独立元素，供特征值导数使用。
    Eigen::Matrix<double, 6, 1> jac;
    jac[0] = u[m][0] * u[n][0];
    jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
    jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
    jac[3] = u[m][1] * u[n][1];
    jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
    jac[5] = u[m][2] * u[n][2];
    return jac;
  }

  void acc_evaluate2(const vector<IMUST>& xs, int head, int end,
                     Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT, double& residual)
  {
    // 累加指定体素范围的最小协方差特征值残差及其一、二阶导数。
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a = head; a < end; a++)
    {
      // 将该体素在各帧中的原始统计量变换到当前位姿下，再合并求总体协方差。
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];

      VOX_FACTOR sig;
      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          sig_tran[i].transform(sig_orig[i], xs[i]);
          sig += sig_tran[i];
        }
      
      const Eigen::Vector3d& vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d& lmbd = saes.eigenvalues();
      const Eigen::Matrix3d& U = saes.eigenvectors();
      int NN = sig.N;
      
      // 最小特征值对应法向方向上的离散程度，作为局部平面残差。
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d& uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i = 0; i < 3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();
          
          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6*i, 0) += jjt;

          const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) +=
            2.0/NN*(combo1-RiTukhat*Pi)*RiTukhat - 2.0/NN/NN*viRiTuk[i]*viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

          Hess.block<6, 6>(6*i, 6*i) += Hb;
        }
      
      for(int i = 0; i < win_size-1; i++)
        if(sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for(int j = i+1; j < win_size; j++)
            if(sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

              Hess.block<6, 6>(6*i, 6*j) += Hb;
            }
        }
      
      residual += lmbd[kk];
    }

    for(int i = 1; i < win_size; i++)
      for(int j = 0; j < i; j++)
        // 上三角块已累加，利用对称性补齐下三角块。
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const vector<IMUST>& xs, double& residual)
  {
    // 只计算当前位姿下所有有效平面体素的残差，用于评估试探步。
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += lmbd[kk];
    }
  }

  std::vector<double> evaluate_residual(const vector<IMUST>& xs)
  {
    // 返回每个体素单独的残差，供按分位数阈值剔除离群约束。
    std::vector<double> residuals;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residuals.push_back(lmbd[kk]);
    }

    return residuals;
  }

  void remove_residual(const vector<IMUST>& xs, double threshold, double reject_num)
  {
    // 从约束列表中删除残差不小于阈值的体素，最多删除 reject_num 个。
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int rej_cnt = 0;
    size_t i = 0;
    for(; i < plvec_voxels.size();)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[i];
      VOX_FACTOR sig;

      for(int j = 0; j < win_size; j++)
      {
        sig_tran[j].transform(sig_orig[j], xs[j]);
        sig += sig_tran[j];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      if(lmbd[kk] >= threshold)
      {
        plvec_voxels.erase(plvec_voxels.begin()+i);
        rej_cnt++;
        continue;
      }
      i++;
      if(rej_cnt == reject_num) break;
    }
  }
};

int BINGO_CNT = 0;
enum OCTO_STATE {UNKNOWN, MID_NODE, PLANE};
// 八叉树节点：保存逐帧点云统计量，并将空间判定为平面、继续细分或停止细分。
class OCTO_TREE_NODE
{
public:
  // 原始/变换后点及其统计量按窗口帧分别存储；子节点对应当前体素的八个象限。
  OCTO_STATE octo_state;
  int layer, win_size;
  vector<PLV(3)> vec_orig, vec_tran;
  vector<VOX_FACTOR> sig_orig, sig_tran;

  OCTO_TREE_NODE* leaves[8];
  float voxel_center[3];
  float quater_length;
  float eigen_thr;

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;
  
  #ifdef ENABLE_RVIZ
  ros::NodeHandle nh;
  ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 1000);
  ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/direct", 1000);
  #endif

  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0/10):
    win_size(_win_size), eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN; layer = 0;
    vec_orig.resize(win_size); vec_tran.resize(win_size);
    sig_orig.resize(win_size); sig_tran.resize(win_size);
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  bool judge_eigen()
  {
    // 合并各帧变换后统计量并做 PCA，以最小特征向量作为候选平面法向。
    VOX_FACTOR covMat;
    for(int i = 0; i < win_size; i++)
      if(sig_tran[i].N > 0)
        covMat += sig_tran[i];
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest
    if(eigen_ratio > eigen_thr) return 0;

    // 再检查八个子区域的最小特征值是否与整块尺度相符，避免把偶然共面误判为稳定平面。
    double eva0 = saes.eigenvalues()[0];
    double sqr_eva0 = sqrt(eva0);
    Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    vector<VOX_FACTOR> covMats(8);
    for(int i = 0; i < win_size; i++)
    {
      for(Eigen::Vector3d ap: vec_tran[i])
      {
        int xyz[3] = {0, 0, 0};
        for(int k = 0; k < 3; k++)
          if(ap(k) > center_turb[k])
            xyz[k] = 1;

        Eigen::Vector3d pvec(ap(0), ap(1), ap(2));
        
        int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
        covMats[leafnum].push(pvec);
      }
    }

    double ratios[2] = {1.0/(3.0*3.0), 2.0*2.0};
    int num_all = 0, num_qua = 0;
    for(int i = 0; i < 8; i++)
      if(covMats[i].N > 10)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
        double child_eva0 = (saes.eigenvalues()[0]);
        if(child_eva0 > ratios[0]*eva0 && child_eva0 < ratios[1]*eva0)
          num_qua++;
        num_all++;
      }

    double prop = 1.0 * num_qua / num_all;

    if(prop < 0.5) return 0;
    return 1;
  }

  void cut_func(int ci)
  {
    // 将第 ci 帧的点按体素中心分到八个子体素，并同步更新子节点统计量。
    PLV(3)& pvec_orig = vec_orig[ci];
    PLV(3)& pvec_tran = vec_tran[ci];

    uint a_size = pvec_tran.size();
    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }

      leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
      leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);
      
      leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
      leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
    }

    PLV(3)().swap(pvec_orig);
    PLV(3)().swap(pvec_tran);
  }

  void recut()
  {
    // UNKNOWN 节点先判断点数和局部平面性；不满足时继续递归细分。
    if(octo_state == UNKNOWN)
    {
      int point_size = 0;
      for(int i = 0; i < win_size; i++)
        point_size += sig_orig[i].N;
      
      if(point_size < MIN_PT)
      {
        // 点数不足以支撑可靠平面估计，将节点标记为内部/终止节点并释放点缓存。
        octo_state = MID_NODE;
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        return;
      }

      if(judge_eigen())
      {
        // 满足平面判据后保留为叶平面，后续可直接生成优化约束。
        octo_state = PLANE;
        #ifndef ENABLE_FILTER
        #ifndef ENABLE_RVIZ
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        #endif
        #endif
        return;
      }
      else
      {
        if(layer == layer_limit)
        {
          // 达到最大细分层数仍不是平面时停止递归。
          octo_state = MID_NODE;
          vector<PLV(3)>().swap(vec_orig);
          vector<PLV(3)>().swap(vec_tran);
          vector<VOX_FACTOR>().swap(sig_orig);
          vector<VOX_FACTOR>().swap(sig_tran);
          return;
        }
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        for(int i = 0; i < win_size; i++)
          cut_func(i);
      }
    }
    
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut();
  }

  void tras_opt(VOX_HESS& vox_opt)
  {
    // 平面叶节点加入 Hessian 约束集合；非叶节点递归访问子节点。
    if(octo_state == PLANE)
      vox_opt.push_voxel(&sig_orig, &vec_orig);
    else
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
  }

  void tras_display(int layer = 0)
  {
    // 可选 RViz 调试输出：发布平面点云和法向箭头，不参与优化计算。
    float ref = 255.0*rand()/(RAND_MAX + 1.0f);
    pcl::PointXYZINormal ap;
    ap.intensity = ref;

    if(octo_state == PLANE)
    {
      // std::vector<unsigned int> colors;
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
      pcl::PointCloud<pcl::PointXYZINormal> color_cloud;

      for(int i = 0; i < win_size; i++)
      {
        for(size_t j = 0; j < vec_tran[i].size(); j++)
        {
          Eigen::Vector3d& pvec = vec_tran[i][j];
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          // ap.b = colors[0];
          // ap.g = colors[1];
          // ap.r = colors[2];
          ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          ap.normal_z = sqrt(value_vector[0]);
          // ap.curvature = total;
          color_cloud.push_back(ap);
        }
      }

      #ifdef ENABLE_RVIZ
      sensor_msgs::PointCloud2 dbg_msg;
      pcl::toROSMsg(color_cloud, dbg_msg);
      dbg_msg.header.frame_id = "camera_init";
      pub_residual.publish(dbg_msg);

      visualization_msgs::Marker marker;
      visualization_msgs::MarkerArray marker_array;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time::now();
      marker.ns = "basic_shapes";
      marker.id = BINGO_CNT; BINGO_CNT++;
      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.color.a = 1;
      marker.color.r = layer==0?1:0;
      marker.color.g = layer==1?1:0;
      marker.color.b = layer==2?1:0;
      marker.scale.x = 0.01;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.lifetime = ros::Duration();
      geometry_msgs::Point apoint;
      apoint.x = center(0); apoint.y = center(1); apoint.z = center(2);
      marker.points.push_back(apoint);
      apoint.x += 0.2*direct(0); apoint.y += 0.2*direct(1); apoint.z += 0.2*direct(2);
      marker.points.push_back(apoint);
      marker_array.markers.push_back(marker);
      pub_direct.publish(marker_array);
      #endif
    }
    else
    {
      if(layer == layer_limit) return;
      layer++;
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(layer);
    }
  }
};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  OCTO_TREE_ROOT(int _winsize, float _eigen_thr): OCTO_TREE_NODE(_winsize, _eigen_thr){}
};

// 窗口位姿优化器：并行构建正规方程，再以 LM 阻尼方式更新各帧位姿。
class VOX_OPTIMIZER
{
public:
  // jac_leng 按每帧 DVEL 维计算；imu_leng 按每帧 DIM 维计算。
  int win_size, jac_leng, imu_leng;
  VOX_OPTIMIZER(int _win_size = WIN_SIZE): win_size(_win_size)
  {
    jac_leng = DVEL * win_size;
    imu_leng = DIM * win_size;
  }

  double divide_thread(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab,
                       Eigen::MatrixXd& Hess,Eigen::VectorXd& JacT)
  {
    // 将体素约束分块交给多个线程计算，最后累加各线程 Hessian、Jacobian 和残差。
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    for(int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for(int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i = 0; i < tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }
    #ifdef AVG_THR
    return residual/g_size;
    #else
    return residual;
    #endif
  }

  double only_residual(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab, bool is_avg = false)
  {
    // 评估给定状态的目标函数值，可选择返回体素平均残差。
    double residual2 = 0;
    voxhess.evaluate_only_residual(x_stats, residual2);
    if(is_avg) return residual2 / voxhess.plvec_voxels.size();
    return residual2;
  }

  void remove_outlier(vector<IMUST>& x_stats, VOX_HESS& voxhess, double ratio)
  {
    // 按残差排序，选择保留比例对应的阈值并移除高残差体素。
    std::vector<double> residuals = voxhess.evaluate_residual(x_stats);
    std::sort(residuals.begin(), residuals.end()); // sort in ascending order
    double threshold = residuals[std::floor((1-ratio)*voxhess.plvec_voxels.size())-1];
    int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
    // std::cout << "vox_num before " << voxhess.plvec_voxels.size();
    // std::cout << ", reject threshold " << std::setprecision(3) << threshold << ", rejected " << reject_num;
    voxhess.remove_residual(x_stats, threshold, reject_num);
    // std::cout << ", vox_num after " << voxhess.plvec_voxels.size() << std::endl;
  }

  void damping_iter(vector<IMUST>& x_stats, VOX_HESS& voxhess, double& residual,
                    PLV(6)& hess_vec, size_t& mem_cost)
  {
    // Levenberg-Marquardt 迭代：解 (H + uD) dx = -J，试探更新并按目标下降情况调整阻尼。
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng),
                    HessuD(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng), new_dxi(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp;

    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0];
    // 将全局位姿转换为相邻帧相对位姿，供局部窗口约束计算使用。
    for(int i=1; i<win_size; i++)
    {
      x_ab[i].p = x_stats[i-1].R.transpose() * (x_stats[i].p - x_stats[i-1].p);
      x_ab[i].R = x_stats[i-1].R.transpose() * x_stats[i].R;
    }

    double hesstime = 0;
    double solvtime = 0;
    size_t max_mem = 0;
    double loop_num = 0;
    for(int i = 0; i < 10; i++)
    {
      // 接受上一步后重新计算正规方程；拒绝时增大阻尼并复用现有 Hessian。
      if(is_calc_hess)
      {
        double tm = ros::Time::now().toSec();
        residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);
        hesstime += ros::Time::now().toSec() - tm;
      }

      double tm = ros::Time::now().toSec();
      D.diagonal() = Hess.diagonal();
      HessuD = Hess + u*D;
      double t1 = ros::Time::now().toSec();
      Eigen::SparseMatrix<double> A1_sparse(jac_leng, jac_leng);
      std::vector<Eigen::Triplet<double>> tripletlist;
      for(int a = 0; a < jac_leng; a++)
        for(int b = 0; b < jac_leng; b++)
          if(HessuD(a, b) != 0)
          {
            tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
            //A1_sparse.insert(a, b) = HessuD(a, b);
          }
      A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
      A1_sparse.makeCompressed();
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
      Solver_sparse.compute(A1_sparse);
      size_t temp_mem = check_mem();
      if(temp_mem > max_mem) max_mem = temp_mem;
      dxi = Solver_sparse.solve(-JacT);
      temp_mem = check_mem();
      if(temp_mem > max_mem) max_mem = temp_mem;
      solvtime += ros::Time::now().toSec() - tm;
      // new_dxi = Solver_sparse.solve(-JacT);
      // printf("new solve time cost %f\n",ros::Time::now().toSec() - t1);
      // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
      // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
      // std::cout<<"relative error "<<relative_err<<std::endl;
      // std::cout<<"absolute error "<<absolute_err<<std::endl;
      // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

      x_stats_temp = x_stats;
      // 以 6 维旋转/平移增量更新每帧状态，先放入临时变量以便判断是否接受。
      for(int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
      }

      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
      #ifdef AVG_THR
      residual2 = only_residual(x_stats_temp, voxhess, x_ab, true);
      q1 /= voxhess.plvec_voxels.size();
      #else
      residual2 = only_residual(x_stats_temp, voxhess, x_ab);
      #endif
      residual = residual2;
      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n",
      //        i, residual1, residual2, u, v, q/q1, q1, q);
      loop_num = i+1;
      // if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
      // if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
      // if(double(max_mem/1048576.0) > 2.0) printf("Max mem: %lf\n", double(max_mem/1048576.0));
      
      if(q > 0)
      {
        // 目标函数下降则接受新状态并减小阻尼。
        x_stats = x_stats_temp;
        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        // 目标未下降则拒绝该步并增大阻尼。
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }
      #ifdef AVG_THR
      if((fabs(residual1-residual2)/residual1) < 0.05 || i == 9)
      {
        // 收敛或达到迭代上限后记录峰值内存和帧间 Hessian 对角约束。
        if(mem_cost < max_mem) mem_cost = max_mem;
        for(int j = 0; j < win_size-1; j++)
          for(int k = j+1; k < win_size; k++)
            hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL*j, DVEL*k).diagonal().segment<DVEL>(0));
        break;
      }
      #else
      if(fabs(residual1-residual2)<1e-9) break;
      #endif
    }
  }

  size_t check_mem()
  {
    // 从 Linux /proc/self/status 读取当前进程常驻内存 VmRSS，单位为 kB。
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while(fgets(line, 128, file) != nullptr)
    {
      if(strncmp(line, "VmRSS:", 6) == 0)
      {
        int len = strlen(line);

        const char* p = line;
        for(; std::isdigit(*p) == false; ++p){}

        line[len - 3] = 0;
        result = atoi(p);

        break;
      }
    }
    fclose(file);

    return result;
  }
};

#endif