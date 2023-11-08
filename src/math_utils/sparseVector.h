#include"vector.h"

#include <iostream>

#ifndef __SPARSEVECTOR__
#define __SPARSEVECTOR__

class Vector;

class SparseVector{
 private:
  //本当のサイズ
  int size_;
  //非ゼロ成分のみサイズ
  int nnz_;
  //非ゼロ成分本当のデータ番号
  int *indices_;
  //非ゼロ成分要素
  double *values_;

 public:
  //コンストラクタ
  SparseVector(int cols_=0, int nnz_=0);
  //コピーコンストラクタ
  SparseVector(const SparseVector &arg);
  //ムーブコンストラクタ
  SparseVector(SparseVector &&arg);
  //デストラクタ
  ~SparseVector(void);
  //コピー代入
  SparseVector &operator=(const SparseVector &arg);
  //ムーブ代入
  SparseVector &operator=(SparseVector &&arg);
  //Sizeを返す
  int size(void) const;
  //nnz_を返す
  int nnz(void) const;
  //Index[index]を返す
  int indexIndex(int index) const;
  //Index[index]を変更する
  int &indexIndex(int index);
  //values_Index[index]を返す
  double values_Index(int index) const;
  //values_Index[index]を変える
  double &values_Index(int index);
  SparseVector operator+(void) const;
  SparseVector operator-(void) const;
  bool operator==(const SparseVector &rhs) const;
  bool operator!=(const SparseVector &rhs) const;
  void modifyvalues_(int n,int index, double value); 
};

std::ostream &operator<<(std::ostream &os, const SparseVector &rhs);
//絶対値をとった要素の最大
double max_norm(const SparseVector &arg);
//2ノルム
double squared_norm(const SparseVector &arg);
//2ノルムの二乗
double norm_square(const SparseVector &arg);
//SparseVectorとVectorの内積
double operator*(const SparseVector &lhs, const Vector &rhs);
double operator*(const Vector &lhs, const SparseVector &rhs);

#endif
