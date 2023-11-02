#include "fm_base.h"

FMBase::FMBase(int missing_pattern) : Recom(missing_pattern) {}

double FMBase::predict_y(Vector &x, double w0, Vector &w, Matrix &v) {
    int n = x.size();  // 非ゼロデータの数
    int k = v.cols();  // 因子数

    double sum_vixi = 0.0;
    for (int f = 0; f < k; ++f) {
        double sum_vx = 0.0;
        double sum_v2x2 = 0.0;
        for (int j = 0; j < n; ++j) {
            if (x[j] != 0) {
                double vx = v[j][f] * x[j];
                sum_vx += vx;
                sum_v2x2 += vx * vx;
            }
        }
        sum_vixi += sum_vx * sum_vx - sum_v2x2;
    }
    double result = w0;
    for (int j = 0; j < n; ++j) {
        if (x[j] != 0) {
            result += x[j] * w[j];
        }
    }
    result += 0.5 * sum_vixi;
    return result;
}