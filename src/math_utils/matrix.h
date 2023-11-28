#include "vector.h"

#ifndef __MATRIX__
#define __MATRIX__

class Matrix {
   private:
    int rows_;
    int cols_;
    double *values_;

   public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double arg);
    Matrix(void);
    Matrix(const Matrix &arg);
    Matrix &operator=(const Matrix &rhs);
    ~Matrix(void);
    int rows(void) const;
    int cols(void) const;
    double &operator()(int row, int col);
    double operator()(int row, int col) const;
    Vector operator[](int row);
    Matrix operator+(void) const;
    Matrix operator-(void) const;
    Matrix &operator+=(const Matrix &rhs);
    Matrix &operator-=(const Matrix &rhs);
    std::ostream &print(std::ostream &lhs) const;
    double *get_values();  // データへのポインタを取得するメソッド
};

std::ostream &operator<<(std::ostream &lhs, const Matrix &rhs);
Matrix operator+(const Matrix &lhs, const Matrix &rhs);
Matrix operator-(const Matrix &lhs, const Matrix &rhs);
Vector operator*(const Matrix &lhs, const Vector &rhs);
Matrix operator*(Matrix &lhs, Matrix &rhs);
bool operator==(const Matrix &lhs, const Matrix &rhs);
bool operator!=(const Matrix &lhs, const Matrix &rhs);
Matrix operator*(double factor, const Matrix &rhs);
double squared_sum(const Matrix &arg);
double frobenius_norm(const Matrix &arg);
Matrix transpose(const Matrix &arg);

#endif
