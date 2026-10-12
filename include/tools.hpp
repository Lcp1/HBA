#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <math.h>

#define HASH_P 116101
#define MAX_N 10000000019
#define SMALL_EPS 1e-10
#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0
#define PLM(a) vector<Eigen::Matrix<double, a, a>, Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define PLV(a) vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>
#define VEC(a) Eigen::Matrix<double, a, 1>

#define G_m_s2 9.81
#define DIMU 18
#define DIM 15
#define DNOI 12
#define NMATCH 5
#define DVEL 6

// IMU 状态维数、位姿维数、噪声维数及点云匹配相关的编译期常量。
typedef pcl::PointXYZ PointType;
// typedef pcl::PointXYZI PointType;
// typedef pcl::PointXYZINormal PointType;
using namespace std;

Eigen::Matrix3d I33(Eigen::Matrix3d::Identity());
Eigen::Matrix<double, DIMU, DIMU> I_imu(Eigen::Matrix<double, DIMU, DIMU>::Identity());

class VOXEL_LOC
{
public:
  // 三维整数体素坐标，用作 unordered_map 的键。
  int64_t x, y, z;

  // 用三个整数坐标构造体素键；未提供参数时表示原点体素。
  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0): x(vx), y(vy), z(vz){}

  // 判断三个坐标是否都相同，供 unordered_map 判断键是否相等。
  bool operator == (const VOXEL_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};

namespace std
{
  // 为体素坐标提供哈希函数，使体素可存入 unordered_map。
  template<>
  struct hash<VOXEL_LOC>
  {
    // 将三维体素坐标混合成哈希值，供 std::unordered_map 使用。
    size_t operator() (const VOXEL_LOC &s) const
    {
      using std::size_t; using std::hash;
      // return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
      long long index_x, index_y, index_z;
			double cub_len = 0.125;
			index_x = int(round(floor((s.x)/cub_len + SMALL_EPS)));
			index_y = int(round(floor((s.y)/cub_len + SMALL_EPS)));
			index_z = int(round(floor((s.z)/cub_len + SMALL_EPS)));
			return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
    }
  };
}

// 返回矩阵元素绝对值之和；用于对矩阵量级进行简单汇总。
// 输入为任意尺寸的稠密矩阵，返回所有元素绝对值的总和。
double matrixAbsSum(Eigen::MatrixXd mat)
{
  double sum = 0.0;
  // 外层遍历行、内层遍历列，将每个矩阵元素的绝对值累加。
  for (int i = 0; i < mat.rows(); i++)
    for (int j = 0; j < mat.cols(); j++)
      sum += fabs(mat(i, j));
  return sum;
}

// Logistic 映射，将实数输入压缩到 (0, 1)。
// 参数 r 为标量输入，返回 1/(1+exp(-r))。
double sigmoid_w(double r)
{
  return 1.0/(1+exp(-r));
}

// SO(3) 指数映射：将旋转向量转换成旋转矩阵。
// 输入向量的方向表示旋转轴、模长表示旋转角（弧度）。
Eigen::Matrix3d Exp(const Eigen::Vector3d &ang)
{
  double ang_norm = ang.norm();
  Eigen::Matrix3d Eye3 = Eigen::Matrix3d::Identity();
  if (ang_norm > 0.0000001)
  {
    Eigen::Vector3d r_axis = ang / ang_norm;
    Eigen::Matrix3d K;
    K << SKEW_SYM_MATRX(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  }
  else
  {
    return Eye3;
  }
}

// 将角速度积分 dt 秒并通过 Rodrigues 公式得到对应旋转矩阵。
// 旋转角为角速度模长乘以时间间隔，旋转轴沿角速度方向。
Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt)
{
  double ang_vel_norm = ang_vel.norm();
  Eigen::Matrix3d Eye3 = Eigen::Matrix3d::Identity();

  if (ang_vel_norm > 0.0000001)
  {
    Eigen::Vector3d r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix3d K;

    K << SKEW_SYM_MATRX(r_axis);
    double r_ang = ang_vel_norm * dt;

    /// Roderigous Tranformation
    return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  }
  else
  {
    return Eye3;
  }
}

// SO(3) 对数映射：从旋转矩阵恢复旋转向量。
// 返回向量方向为旋转轴、模长为旋转角（弧度）。
Eigen::Vector3d Log(const Eigen::Matrix3d &R)
{
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Vector3d K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

// 向量的反对称矩阵（hat 算子），满足 hat(v) * w = v × w。
Eigen::Matrix3d hat(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d Omega;
  Omega <<  0, -v(2),  v(1)
      ,  v(2),     0, -v(0)
      , -v(1),  v(0),     0;
  return Omega;
}

// 根据旋转向量计算 SO(3) 右雅可比，用于旋转扰动的局部线性化。
Eigen::Matrix3d jr(Eigen::Vector3d vec)
{
  double ang = vec.norm();

  if(ang < 1e-9)
  {
    return I33;
  }
  else
  {
    vec /= ang;
    double ra = sin(ang)/ang;
    return ra*I33 + (1-ra)*vec*vec.transpose() - (1-cos(ang))/ang * hat(vec);
  }
}

// 计算旋转矩阵对应旋转向量处的 SO(3) 右雅可比逆。
Eigen::Matrix3d jr_inv(const Eigen::Matrix3d &rotR)
{
  Eigen::AngleAxisd rot_vec(rotR);
  Eigen::Vector3d axi = rot_vec.axis();
  double ang = rot_vec.angle();

  if(ang < 1e-9)
  {
    return I33;
  }
  else
  {
    double ctt = ang / 2 / tan(ang/2);
    return ctt*I33 + (1-ctt)*axi*axi.transpose() + ang/2 * hat(axi);
  }
}

struct IMUST
{
  // 时间、姿态、位置、速度、陀螺仪/加速度计偏置和重力向量。
  double t;
  Eigen::Matrix3d R;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  Eigen::Vector3d g;
  
  // 初始化为默认误差状态：单位旋转、零向量状态和标准重力。
  IMUST()
  {
    setZero();
  }

  // 按给定时间、姿态、位置、速度和传感器偏置构造状态；重力参数可省略。
  IMUST(double _t, const Eigen::Matrix3d& _R, const Eigen::Vector3d& _p, const Eigen::Vector3d& _v,
        const Eigen::Vector3d& _bg, const Eigen::Vector3d& _ba,
        const Eigen::Vector3d& _g = Eigen::Vector3d(0, 0, -G_m_s2)):
        t(_t), R(_R), p(_p), v(_v), bg(_bg), ba(_ba), g(_g){}

  // 将 18 维误差状态加到当前状态；姿态右乘 Exp(旋转增量)，其余字段逐项相加。
  IMUST &operator+=(const Eigen::Matrix<double, DIMU, 1> &ist)
  {
    this->R = this->R * Exp(ist.block<3, 1>(0, 0));
    this->p += ist.block<3, 1>(3, 0);
    this->v += ist.block<3, 1>(6, 0);
    this->bg += ist.block<3, 1>(9, 0);
    this->ba += ist.block<3, 1>(12, 0);
    this->g += ist.block<3, 1>(15, 0);
    return *this;
  }

  // 计算当前状态相对 b 的 18 维误差向量，不修改两个状态对象。
  Eigen::Matrix<double, DIMU, 1> operator-(const IMUST &b) 
  {
    // 姿态误差用 SO(3) 对数映射表示，其余状态按分量相减。
    Eigen::Matrix<double, DIMU, 1> a;
    a.block<3, 1>(0, 0) = Log(b.R.transpose() * this->R);
    a.block<3, 1>(3, 0) = this->p - b.p;
    a.block<3, 1>(6, 0) = this->v - b.v;
    a.block<3, 1>(9, 0) = this->bg - b.bg;
    a.block<3, 1>(12, 0) = this->ba - b.ba;
    a.block<3, 1>(15, 0) = this->g - b.g;
    return a;
  }

  // 将 b 的所有状态字段复制到当前对象，返回当前对象引用以支持连续赋值。
  IMUST &operator=(const IMUST &b)
  {
    this->R = b.R;
    this->p = b.p;
    this->v = b.v;
    this->bg = b.bg;
    this->ba = b.ba;
    this->g = b.g;
    this->t = b.t;
    return *this;
  }

  // 将状态重置为默认值；时间归零，重力设为 (0, 0, -G_m_s2)。
  void setZero()
  {
    // 初始化为单位姿态、零位置/速度/偏置和标准向下重力。
    t = 0; R.setIdentity();
    p.setZero(); v.setZero();
    bg.setZero(); ba.setZero();
    g << 0, 0, -G_m_s2;
  }
};
  // 将输入四元数和平移复制到输出引用，不执行归一化或坐标变换。

void assign_qt(Eigen::Quaterniond& q, Eigen::Vector3d& t,
               const Eigen::Quaterniond& q_, const Eigen::Vector3d& t_)
{
  q.w() = q_.w(); q.x() = q_.x(); q.y() = q_.y(); q.z() = q_.z();
  t(0) = t_(0); t(1) = t_(1); t(2) = t_(2);
}

// 一个体素内的点坐标和数量，用于累计点云质心。
struct M_POINT
{
	float xyz[3];
	int count = 0;
};

void downsample_voxel(pcl::PointCloud<PointType>& pc, double voxel_size)
{
  // 原地替换点云内容；voxel_size 是体素边长，小于 0.01 时保持输入不变。
  // 每个非空体素以点坐标均值作为代表点；小于 1 cm 时不执行降采样。
	if (voxel_size < 0.01)
		return;

	std::unordered_map<VOXEL_LOC, M_POINT> feature_map;
	size_t pt_size = pc.size();

  // 逐点计算体素索引；同一索引的点累加到同一个 M_POINT 中。
	for (size_t i = 0; i < pt_size; i++)
	{
    // 通过整数体素坐标聚合点；负坐标减 1 用于避免一般负小数向零截断。
		PointType &pt_trans = pc[i];
		float loc_xyz[3];
    // x/y/z 分别除以体素边长，生成三维体素坐标。
		for (int j = 0; j < 3; j++)
		{
			loc_xyz[j] = pt_trans.data[j] / voxel_size;
			if (loc_xyz[j] < 0)
				loc_xyz[j] -= 1.0;
		}

		VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
		auto iter = feature_map.find(position);
		if (iter != feature_map.end())
		{
			iter->second.xyz[0] += pt_trans.x;
			iter->second.xyz[1] += pt_trans.y;
			iter->second.xyz[2] += pt_trans.z;
			iter->second.count++;
		}
		else
		{
			M_POINT anp;
			anp.xyz[0] = pt_trans.x;
			anp.xyz[1] = pt_trans.y;
			anp.xyz[2] = pt_trans.z;
			anp.count = 1;
			feature_map[position] = anp;
		}
	}

	pt_size = feature_map.size();
	pc.clear();
	pc.resize(pt_size);

  // 用各体素点的算术平均值替换原始点云。
	size_t i = 0;
  // 遍历所有非空体素，将坐标总和除以点数写回压缩后的点云。
	for (auto iter = feature_map.begin(); iter != feature_map.end(); ++iter)
	{
		pc[i].x = iter->second.xyz[0] / iter->second.count;
		pc[i].y = iter->second.xyz[1] / iter->second.count;
		pc[i].z = iter->second.xyz[2] / iter->second.count;
		i++;
	}
}

// 对点云原地应用刚体变换 p' = R p + t。
void pl_transform(pcl::PointCloud<PointType> &pl1, const Eigen::Matrix3d &rr, const Eigen::Vector3d &tt)
{
  // 原地逐点应用旋转矩阵 rr 和平移向量 tt。
  // 每次循环只更新一个点的 XYZ，不改变点云点数。
  for(PointType &ap : pl1.points)
  {
    Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
    pvec = rr * pvec + tt;
    ap.x = pvec[0];
    ap.y = pvec[1];
    ap.z = pvec[2];
  }
}

// 将 IMU 位姿应用到一组 3D 向量，输出变换后的坐标。
void plvec_trans(PLV(3) &porig, PLV(3) &ptran, IMUST &stat)
{
  // 将输入 3D 向量数组按 stat.R/stat.p 变换，并调整输出数组长度与输入一致。
  uint asize = porig.size();
  ptran.resize(asize);
  // 按索引一一对应地执行 R*p+t，输入 porig 保持不变。
  for(uint i=0; i<asize; i++)
    ptran[i] = stat.R * porig[i] + stat.p;
}

// bool time_compare(PointType &x, PointType &y) {return (x.curvature < y.curvature);}

class VOX_FACTOR
{
public:
  // P = sum(p p^T)、v = sum(p)、N = 点数；三者可合并并恢复协方差。
  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  int N;

  // 创建空统计量：二阶矩、坐标和与点数均清零。
  VOX_FACTOR()
  {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void clear()
  {
    // 清空当前累计值，使该对象可复用于新的点集。
    P.setZero();
    v.setZero();
    N = 0;
  }

  void push(const Eigen::Vector3d &vec)
  {
    // 将一个三维点累加到二阶矩、坐标和及点数中。
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d cov()
  {
    // 根据当前累计量计算总体协方差；调用前要求 N 大于零。
    // 使用二阶矩减去均值外积，得到总体协方差矩阵。
    Eigen::Vector3d center = v / N;
    return P/N - center*center.transpose();
  }

  VOX_FACTOR & operator+=(const VOX_FACTOR& sigv)
  {
    // 合并另一个点集的充分统计量，返回当前对象引用。
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }

  void transform(const VOX_FACTOR &sigv, const IMUST &stat)
  {
    // 将 sigv 的统计量按 stat 的刚体位姿解析变换并写入当前对象。
    // 将聚合统计量按刚体变换解析更新，避免逐点变换后重新累加。
    N = sigv.N;
    v = stat.R*sigv.v + N*stat.p;
    Eigen::Matrix3d rp = stat.R * sigv.v * stat.p.transpose();
    P = stat.R*sigv.P*stat.R.transpose() + rp + rp.transpose() + N*stat.p*stat.p.transpose();
  }
};

const double threshold = 0.1;
// 用固定数量的点拟合平面 ax+by+cz+1=0，并检查各点的平面残差。
// 成功时将归一化平面系数写入 pca_result 并返回 true，否则返回 false。
bool esti_plane(Eigen::Vector4d &pca_result, const pcl::PointCloud<PointType> &point)
{
  Eigen::Matrix<double, NMATCH, 3> A;
  Eigen::Matrix<double, NMATCH, 1> b;
  b.setOnes();
  b *= -1.0f;

  // 将前 NMATCH 个点的坐标作为线性方程系数，求解平面法向的未归一化参数。
  for (int j = 0; j < NMATCH; j++) 
  {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);

  // 检查参与拟合的每个点到候选平面的代数残差，任一点超阈值则判定失败。
  for (int j = 0; j < NMATCH; j++) 
  {
    if (fabs(normvec.dot(A.row(j)) + 1.0) > threshold) 
      return false;
  }

  double n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;
  return true;
}

#endif
