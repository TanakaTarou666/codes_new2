#include "fm_base.h"

FMBase::FMBase(int missing_pattern) : Recom(missing_pattern) {}

double FMBase::predict_y(SparseVector &x, double w0, Vector w, Matrix &v) {
    int n = x.nnz();   // 非ゼロデータの数
    int k = v.rows();  // 因子数

    double sum_vixi = 0.0;
    for (int f = 0; f < k; ++f) {
        double sum_vx = 0.0;
        double sum_v2x2 = 0.0;
        for (int j = 0; j < n; ++j) {
            double vx = v(f, x.dense_index(j)) * x(j);
            sum_vx += vx;
            sum_v2x2 += vx * vx;
        }
        sum_vixi += sum_vx * sum_vx - sum_v2x2;
    }
    double result = w0;
    for (int j = 0; j < n; ++j) {
        result += x(j) * w[x.dense_index(j)];
    }
    result += 0.5 * sum_vixi;
    return result;
}

SparseVector FMBase::make_one_hot_data(int user_index, int item_index) {
    SparseVector result(rs::num_users + rs::num_items, 2);
    result(0) = 1;
    result.dense_index(0) = user_index;
    result(1) = 1;
    result.dense_index(1) = rs::num_users + item_index;
    return result;
}
