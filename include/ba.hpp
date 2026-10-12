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

  // 创建窗口级约束容器，并为每帧初始化可选的原始点缓存。
  VOX_HESS(int _win_size = WIN_SIZE): win_size(_win_size){origin_points.resize(win_size);}

  // 释放体素指针列表占用的容器存储；体素节点本身由八叉树拥有并释放。
  ~VOX_HESS()
  {
    vector<const vector<VOX_FACTOR>*>().swap(plvec_voxels);
  }

  // 将 vec_orig 中的点追加到 origin_points_，供可选过滤或调试流程使用。
  void get_center(const PLV(3)& vec_orig, PLV(3)& origin_points_)
  {
    // 将体素内点坐标复制到调试缓存。
    size_t pt_size = vec_orig.size();
    // 顺序复制该帧该体素中的每个三维点。
    for(size_t i = 0; i < pt_size; i++)
      origin_points_.emplace_back(vec_orig[i]);
    return;
  }

  // 注册一个体素的逐帧统计量；少于两帧观测到的体素不作为跨帧约束。
  // sig_orig 和 vec_orig 指向八叉树节点持有的数据，必须比本对象活得更久。
  void push_voxel(const vector<VOX_FACTOR>* sig_orig, const vector<PLV(3)>* vec_orig)
  {
    int process_size = 0;
    // 统计当前窗口中实际观测到该体素的帧数。
    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        process_size++;

    #ifdef ENABLE_FILTER
    if(process_size < 1) return;

    // 若启用调试过滤，复制每个有观测帧的原始点到 origin_points。
    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        get_center((*vec_orig)[i], origin_points[i]);
    #endif
    
    if(process_size < 2) return;
    
    plvec_voxels.push_back(sig_orig);
  }

  // 生成 u[m]u[n]^T 的六个独立对称元素，供特征值微分中的矩阵向量化使用。
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

  // 对体素索引 [head, end) 累加目标残差、Hessian 和梯度向量 JacT。
  // xs 是当前窗口位姿；三个输出引用在函数入口清零后写入本次分块结果。
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

    // 外层按线程分配的 [head, end) 体素区间遍历，每个体素独立贡献残差和导数。
    for(int a = head; a < end; a++)
    {
      // 将该体素在各帧中的原始统计量变换到当前位姿下，再合并求总体协方差。
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];

      VOX_FACTOR sig;
      // 将该体素在窗口各帧中的统计量变换到世界坐标，再合并为总体统计量。
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
      // 累加另外两个特征方向对最小特征值导数的影响。
      for(int i = 0; i < 3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      // 对每个观测该体素的帧计算 6 维位姿 Jacobian 和自身 Hessian 对角块。
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
      
      // 枚举帧间无序组合 i<j，累加 Hessian 的非对角交叉块。
      for(int i = 0; i < win_size-1; i++)
        if(sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          // j 从 i+1 开始，只处理每对帧一次，并跳过未观测该体素的帧。
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

    // Hessian 上三角已累加，按块转置填充所有下三角位置。
    for(int i = 1; i < win_size; i++)
      // 每行只处理对角线左侧的块，复制对应转置块补齐对称矩阵。
      for(int j = 0; j < i; j++)
        // 上三角块已累加，利用对称性补齐下三角块。
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  // 计算所有已注册体素的残差总和，写入 residual；不计算导数。
  void evaluate_only_residual(const vector<IMUST>& xs, double& residual)
  {
    // 只计算当前位姿下所有有效平面体素的残差，用于评估试探步。
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    // 对所有有效体素逐个重算合并协方差，并把最小特征值累加到总目标函数。
    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      // 合并该体素跨窗口帧的变换后统计量。
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

  // 返回每个已注册体素各自的最小协方差特征值残差，顺序与 plvec_voxels 对应。
  std::vector<double> evaluate_residual(const vector<IMUST>& xs)
  {
    // 返回每个体素单独的残差，供按分位数阈值剔除离群约束。
    std::vector<double> residuals;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int gps_size = plvec_voxels.size();

    // 逐体素计算一次残差，保留其顺序以便与约束列表对应。
    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      // 把当前体素在各帧中的统计量变换并合并后求特征值。
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

  // 按阈值从约束列表删除高残差体素，最多删除 reject_num 个；会修改 plvec_voxels。
  void remove_residual(const vector<IMUST>& xs, double threshold, double reject_num)
  {
    // 从约束列表中删除残差不小于阈值的体素，最多删除 reject_num 个。
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int rej_cnt = 0;
    size_t i = 0;
    // 当前索引只有在保留约束时才递增；删除元素后原索引位置已是下一条约束。
    for(; i < plvec_voxels.size();)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[i];
      VOX_FACTOR sig;

      // 合并当前候选体素在整个窗口内的统计量以重新计算残差。
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

  // 初始化未知节点，为窗口内每帧分配点/统计量容器，并清空八个子节点指针。
  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0/10):
    win_size(_win_size), eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN; layer = 0;
    vec_orig.resize(win_size); vec_tran.resize(win_size);
    sig_orig.resize(win_size); sig_tran.resize(win_size);
    // 初始时八个象限都没有子节点，后续按点的空间位置按需创建。
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  // 递归删除所有已创建子节点，释放整棵子树。
  virtual ~OCTO_TREE_NODE()
  {
    // 递归释放所有已分配子节点；空指针表示该象限从未创建。
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  // PCA 判定当前节点是否为有效平面；更新中心、法向、特征值和特征值比例。
  // 返回 true 表示通过阈值及子区域稳定性检查。
  bool judge_eigen()
  {
    // 合并各帧变换后统计量并做 PCA，以最小特征向量作为候选平面法向。
    VOX_FACTOR covMat;
    // 汇总窗口中有点落入当前节点的帧，空帧不参与协方差。
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
    // 遍历每帧当前节点的变换后点，按扰动后的中心划分到 8 组以评估子区域平面稳定性。
    for(int i = 0; i < win_size; i++)
    {
      // 当前帧逐点检查三个坐标轴，构造该点对应的八分区编号。
      for(Eigen::Vector3d ap: vec_tran[i])
      {
        int xyz[3] = {0, 0, 0};
        // 每个坐标高于中心扰动阈值时置对应二进制位。
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
    // 仅检查点数大于 10 的子区，并统计其最小特征值落入设定比例范围的数量。
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

  // 将窗口中第 ci 帧的点按当前节点中心分配到八个子节点，并转移对应统计量。
  // 完成后释放当前节点该帧的点坐标缓存。
  void cut_func(int ci)
  {
    // 将第 ci 帧的点按体素中心分到八个子体素，并同步更新子节点统计量。
    PLV(3)& pvec_orig = vec_orig[ci];
    PLV(3)& pvec_tran = vec_tran[ci];

    uint a_size = pvec_tran.size();
    // 将当前帧的每个点按当前节点中心分配到唯一子节点。
    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      // x/y/z 三个比较结果编码为 0 到 7 的子节点编号。
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

  // 判断未知节点、必要时递归八分；更新 octo_state 并按节点状态释放不再需要的缓存。
  void recut()
  {
    // UNKNOWN 节点先判断点数和局部平面性；不满足时继续递归细分。
    if(octo_state == UNKNOWN)
    {
      int point_size = 0;
      // 汇总当前节点跨窗口所有帧的点数，用于判断是否达到 MIN_PT。
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
        // 点数不足后不再保留子节点数据，直接结束当前节点处理。
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
        // 对窗口的每一帧分别切分点集，子节点沿用相同帧索引。
        for(int i = 0; i < win_size; i++)
          cut_func(i);
      }
    }
    
    // 当前节点判定完成后继续处理已创建的子树，直到访问所有叶节点。
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut();
  }

  // 将本节点的平面统计量加入 vox_opt；若本节点不是平面，则递归访问子节点。
  void tras_opt(VOX_HESS& vox_opt)
  {
    // 平面叶节点加入 Hessian 约束集合；非叶节点递归访问子节点。
    if(octo_state == PLANE)
      vox_opt.push_voxel(&sig_orig, &vec_orig);
    else
      // 非平面节点递归访问已存在的 8 个子树，不创建空分支。
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
  }

  // 调试用递归可视化接口：发布平面点云和法向箭头，依赖 ENABLE_RVIZ 编译开关。
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

      // 遍历窗口中的每一帧，再遍历该帧保留在当前平面叶节点中的点。
      for(int i = 0; i < win_size; i++)
      {
        // 每个点发布为带特征值尺度信息的调试点。
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
      // 调试显示沿八叉树向下递归，仅处理已经创建的子节点。
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(layer);
    }
  }
};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  // 创建八叉树根节点，参数传递给基类以确定窗口帧数和平面特征值阈值。
  OCTO_TREE_ROOT(int _winsize, float _eigen_thr): OCTO_TREE_NODE(_winsize, _eigen_thr){}
};

// 窗口位姿优化器：并行构建正规方程，再以 LM 阻尼方式更新各帧位姿。
class VOX_OPTIMIZER
{
public:
  // jac_leng 按每帧 DVEL 维计算；imu_leng 按每帧 DIM 维计算。
  int win_size, jac_leng, imu_leng;
  // 按窗口帧数初始化位姿变量总维度和 IMU 状态总维度。
  VOX_OPTIMIZER(int _win_size = WIN_SIZE): win_size(_win_size)
  {
    jac_leng = DVEL * win_size;
    imu_leng = DIM * win_size;
  }

  // 将体素范围分给多个线程并行计算正规方程，归并 Hessian/Jacobian。
  // 返回残差总和；定义 AVG_THR 时返回体素平均残差。
  double divide_thread(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab,
                       Eigen::MatrixXd& Hess,Eigen::VectorXd& JacT)
  {
    // 将体素约束分块交给多个线程计算，最后累加各线程 Hessian、Jacobian 和残差。
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    // 为每个配置线程分配独立正规方程缓冲区，避免并行写入同一 Hessian/梯度。
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
    // 将有效体素索引区间近似均分，启动线程分别计算各自的残差和导数。
    for(int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // 逐线程等待完成并归并局部 Hessian、梯度和残差，然后释放线程对象。
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

  // 仅评估给定位姿的目标函数；is_avg 为 true 时返回每个有效体素的平均残差。
  double only_residual(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab, bool is_avg = false)
  {
    // 评估给定状态的目标函数值，可选择返回体素平均残差。
    double residual2 = 0;
    voxhess.evaluate_only_residual(x_stats, residual2);
    if(is_avg) return residual2 / voxhess.plvec_voxels.size();
    return residual2;
  }

  // 按 ratio 指定的高残差比例确定阈值，并从 voxhess 中移除对应离群体素。
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

  // 对 x_stats 原地执行 LM 位姿迭代；输出最终残差、帧间 Hessian 对角摘要和峰值内存。
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
    // 第一帧作为基准；其余帧依次转换到各自前一帧的相对坐标表示。
    for(int i=1; i<win_size; i++)
    {
      x_ab[i].p = x_stats[i-1].R.transpose() * (x_stats[i].p - x_stats[i-1].p);
      x_ab[i].R = x_stats[i-1].R.transpose() * x_stats[i].R;
    }

    double hesstime = 0;
    double solvtime = 0;
    size_t max_mem = 0;
    double loop_num = 0;
    // 最多进行 10 次 LM 试探；接受或拒绝每一步时动态调整阻尼系数。
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
      // 扫描阻尼 Hessian 的所有元素，仅把非零项转成稀疏矩阵 triplet。
      for(int a = 0; a < jac_leng; a++)
        // 内层遍历当前行的列坐标，保留非零矩阵元素。
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
  // 每帧占 DVEL 个变量：前三项是旋转增量，后三项是平移增量。
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
        // 为每一对不同帧提取 6 维位姿 Hessian 交叉块的对角线，供后续位姿图使用。
        for(int j = 0; j < win_size-1; j++)
          // k 始终大于 j，因此每对帧只保存一次且与上三角块顺序一致。
          for(int k = j+1; k < win_size; k++)
            hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL*j, DVEL*k).diagonal().segment<DVEL>(0));
        break;
      }
      #else
      if(fabs(residual1-residual2)<1e-9) break;
      #endif
    }
  }

  // 读取当前进程的 Linux VmRSS 常驻内存值，返回单位为 kB 的整数读数。
  size_t check_mem()
  {
    // 从 Linux /proc/self/status 读取当前进程常驻内存 VmRSS，单位为 kB。
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    // 逐行扫描进程状态文件，只需找到 VmRSS 字段即可停止。
    while(fgets(line, 128, file) != nullptr)
    {
      if(strncmp(line, "VmRSS:", 6) == 0)
      {
        int len = strlen(line);

        const char* p = line;
  // 跳过字段名和空白，定位 RSS 数值的起始字符。
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