#include "fm_base.h"

FMBase::FMBase(int missing_pattern) : Recom(missing_pattern) {}

double FMBase::predict_y(SparseVector &x, double w0, Vector w, Matrix &v) {
    int n = x.nnz();  // 非ゼロデータの数
    int k = v.cols();  // 因子数

    double sum_vixi = 0.0;
    for (int f = 0; f < k; ++f) {
        double sum_vx = 0.0;
        double sum_v2x2 = 0.0;
        for (int j = 0; j < n; ++j) {
                double vx = v[x(j,"index")][f] * x(j);
                sum_vx += vx;
                sum_v2x2 += vx * vx;
        }
        sum_vixi += sum_vx * sum_vx - sum_v2x2;
    }
    double result = w0;
    for (int j = 0; j < n; ++j) {
            result += x(j) * w[x(j,"index")];
    }
    result += 0.5 * sum_vixi;
    return result;
}

void FMBase::train() {
    int error_count = 0;
    double best_objective_value = DBL_MAX;
    for (int initial_value_index = 0; initial_value_index < num_initial_values; initial_value_index++) {
        std::cout << method_name_ << ": initial setting " << initial_value_index << std::endl;
        set_initial_values(initial_value_index);
        error_detected_ = false;
#ifndef ARTIFICIALITY
        prev_objective_value_ = DBL_MAX;
#endif
        precompute();
        for (int step = 0; step < steps; step++) {
            calculate_factors();
            std::cout << "step:" << step << "\t"
                      << "objective_value:" << calculate_objective_value() << "\t";
            std::cout << std::endl;
            // 収束条件
            if (calculate_convergence_criterion()) {
                break;
            }
            if (step == steps - 1) {
                error_detected_ = true;
                break;
            }
        }

        if (error_detected_) {
            error_count++;
            // 初期値全部{NaN出た or step上限回更新して収束しなかった} =>
            if (error_count == num_initial_values) {
                return;
            }
        } else {
            double objective_value = calculate_objective_value();
            if (objective_value < best_objective_value) {
                best_objective_value = objective_value;
                calculate_prediction();
            }
        }
    }
}

void FMBase::precompute(){return;}
