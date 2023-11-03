#include "tensor.h"

#include <cmath>
#include <iostream>

Tensor::Tensor(int heights, int rows, int cols)
    : heights_(heights), rows_(rows), cols_(cols) {
    matrices_ = new Matrix[heights];
    for (int i = 0; i < heights; i++) {
        matrices_[i] = Matrix(rows, cols);
    }
}

Tensor::Tensor(int heights, int rows, int cols, double arg)
    : heights_(heights), rows_(rows), cols_(cols) {
    matrices_ = new Matrix[heights];
    for (int i = 0; i < heights; i++) {
        matrices_[i] = Matrix(rows, cols, arg);
    }
}

Tensor::Tensor(const Tensor& arg)
    : heights_(arg.heights_), rows_(arg.rows_), cols_(arg.cols_) {
    matrices_ = new Matrix[heights_];
    for (int h = 0; h < heights_; ++h) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                (*this)[h](i, j) = arg[h](i, j);
            }
        }
    }
}

Tensor::Tensor(Tensor& arg)
    : heights_(arg.heights_), rows_(arg.rows_), cols_(arg.cols_) {
    matrices_ = new Matrix[heights_];
    for (int h = 0; h < heights_; ++h) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                (*this)[h](i, j) = arg[h](i, j);
            }
        }
    }
}

Tensor::Tensor() { matrices_ = nullptr; }

// デストラクタ
Tensor::~Tensor(void) { delete[] matrices_; }

int Tensor::heights(void) const { return heights_; }

int Tensor::rows(void) const { return rows_; }

int Tensor::cols(void) const { return cols_; }

Matrix Tensor::operator[](int height) const { return matrices_[height]; }

Matrix& Tensor::operator[](int height) { return matrices_[height]; }

Tensor& Tensor::operator=(const Tensor& arg) {
    if (this == &arg) {
        return *this;  // 自己代入の場合、何もしない
    }

    // メンバー変数をコピー
    heights_ = arg.heights_;
    rows_ = arg.rows_;
    cols_ = arg.cols_;

    // 既存のリソースを解放
    delete[] matrices_;

    // 新しいリソースを確保
    matrices_ = new Matrix[heights_];

    // メンバー変数をコピー
    for (int i = 0; i < heights_; i++) {
        matrices_[i] = arg.matrices_[i];
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& arg) {
    if (this == &arg) {
        return *this;  // 自己代入の場合、何もしない
    }

    // 既存のリソースを解放
    delete[] matrices_;

    // メンバー変数をムーブ
    heights_ = arg.heights_;
    rows_ = arg.rows_;
    cols_ = arg.cols_;
    matrices_ = arg.matrices_;

    // 右辺値のリソースを無効化
    arg.heights_ = 0;
    arg.rows_ = 0;
    arg.cols_ = 0;
    arg.matrices_ = nullptr;

    return *this;
}

Tensor operator+(Tensor& lhs, Tensor& rhs) {
    int heights = lhs.heights();
    int rows = lhs.rows();
    int cols = lhs.cols();

    Tensor result(heights, rows, cols);
    for (int h = 0; h < heights; h++) {
        double* values_A = lhs[0].get_values();
        double* values_B = rhs[0].get_values();
        double* values_Result = result[h].get_values();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                *values_Result++ = *values_A++ + *values_B++;
            }
        }
    }
    return result;
}

Tensor operator-(Tensor& lhs, Tensor& rhs) {
    int heights = lhs.heights();
    int rows = lhs.rows();
    int cols = lhs.cols();

    Tensor result(heights, rows, cols);
    for (int h = 0; h < heights; h++) {
        double* values_A = lhs[0].get_values();
        double* values_B = rhs[0].get_values();
        double* values_Result = result[h].get_values();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                *values_Result++ = *values_A++ - *values_B++;
            }
        }
    }
    return result;
}

double frobenius_norm(const Tensor& arg) {
    double result = 0.0;
    for (int i = 0; i < arg.heights(); i++) {
        result += frobenius_norm(arg[i]);
    }
    return result;
}