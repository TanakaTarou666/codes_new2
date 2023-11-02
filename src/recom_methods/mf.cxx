#include "mf.h"

MF::MF(int missing_pattern) : Recom(missing_pattern), user_factors_(), item_factors_(), prev_user_factors_(), prev_item_factors_() {
    method_name_ = "MFtest32";
}  // ファイル名

void MF::set_parameters(double latent_dimension_percentage, double learning_rate, double reg_parameter) {
#if defined ARTIFICIALITY
    latent_dimension_ = latent_dimension_percentage;
#else
    if (num_users > num_items) {
        latent_dimension_ = std::round(num_items * latent_dimension_percentage / 100);
    } else {
        latent_dimension_ = std::round(num_users * latent_dimension_percentage / 100);
    }
    if (steps < 50) {
        std::cerr << "MF: \"step\" should be 50 or more.";
        return;
    }
#endif
    reg_parameter_ = reg_parameter;
    learning_rate_ = learning_rate;
    parameters_ = {(double)latent_dimension_, reg_parameter_, learning_rate_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
    user_factors_ = Matrix(num_users, latent_dimension_);
    item_factors_ = Matrix(num_items, latent_dimension_);
    return;
}

void MF::train() {  // mf_pred
    int error_count = 0;
    double best_objective_value = DBL_MAX;
    for (int initial_value_index = 0; initial_value_index < num_initial_values; initial_value_index++) {
        std::cout << method_name_ << ": initial setting " << initial_value_index << std::endl;
        set_initial_values(initial_value_index);
        error_detected_ = false;
#ifndef ARTIFICIALITY
        prev_objective_value_ = DBL_MAX;
#endif
        for (int step = 0; step < steps; step++) {
            calculate_user_item_factors();
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
            // 初期値全部{NaN出た or step上限回更新して収束しなかった} => 1を返して終了
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

void MF::set_initial_values(int &seed) {
    std::mt19937_64 mt;
    for (int k_i = 0; k_i < user_factors_.cols(); k_i++) {
        for (int i = 0; i < user_factors_.rows(); i++) {
            mt.seed(seed);
            // ランダムに値生成
            std::uniform_real_distribution<> rand_user_factors_(0.001, 1.0);
            user_factors_[i][k_i] = rand_user_factors_(mt);
            seed++;
        }
        for (int j = 0; j < item_factors_.rows(); j++) {
            mt.seed(seed);
            std::uniform_real_distribution<> rand_item_factors_(0.001, 1.0);
            item_factors_[j][k_i] = rand_item_factors_(mt);
            seed++;
        }
    }
}

void MF::calculate_user_item_factors() {
    prev_item_factors_ = item_factors_;
    prev_user_factors_ = user_factors_;
    user_factor_values_ = user_factors_.get_values();
    for (int i = 0; i < user_factors_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                item_factor_values_ =
                    item_factors_.get_values() + sparse_missing_data_col_indices_[sparse_missing_data_row_pointers_[i] + j] * latent_dimension_;
                // e = r - p * q
                double err = sparse_missing_data_values_[sparse_missing_data_row_pointers_[i] + j];
                for (int k = 0; k < latent_dimension_; k++) {
                    err -= user_factor_values_[k] * item_factor_values_[k];
                }
                for (int k = 0; k < latent_dimension_; k++) {
                    user_factor_values_[k] += learning_rate_ * (2 * err * item_factor_values_[k] - reg_parameter_ * user_factor_values_[k]);
                    item_factor_values_[k] += learning_rate_ * (2 * err * user_factor_values_[k] - reg_parameter_ * item_factor_values_[k]);
                }
            }
        }
        user_factor_values_ += latent_dimension_;
    }
}

double MF::calculate_objective_value() {
    double result = 0.0;
    double P_L2Norm = 0.0, Q_L2Norm = 0.0;
    user_factor_values_ = user_factors_.get_values();
    item_factor_values_ = item_factors_.get_values();
    for (int i = 0; i < user_factors_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                item_factor_values_ =
                    item_factors_.get_values() + sparse_missing_data_col_indices_[sparse_missing_data_row_pointers_[i] + j] * latent_dimension_;
                // e = r - p * q
                double err = sparse_missing_data_values_[sparse_missing_data_row_pointers_[i] + j];
                for (int k = 0; k < latent_dimension_; k++) {
                    err -= user_factor_values_[k] * item_factor_values_[k];
                }
                result += err * err;
            }
        }
        for (int k = 0; k < latent_dimension_; k++) {
            P_L2Norm += user_factor_values_[k] * user_factor_values_[k];
        }
        user_factor_values_ += latent_dimension_;
    }
    item_factor_values_ = item_factors_.get_values();
    for (int j = 0; j < item_factors_.rows(); j++) {
        for (int k = 0; k < latent_dimension_; k++) {
            Q_L2Norm += item_factor_values_[k] * item_factor_values_[k];
        }
        item_factor_values_ += latent_dimension_;
    }
    result += (reg_parameter_ / 2.0) * (P_L2Norm + Q_L2Norm);
    return result;
}

bool MF::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = frobenius_norm((prev_user_factors_ - user_factors_)) + frobenius_norm(prev_item_factors_ - item_factors_);
#else
    objective_value_ = calculate_objective_value();
    double diff = (prev_objective_value_ - objective_value_) / prev_objective_value_;
    prev_objective_value_ = objective_value_;
#endif
    if (std::isfinite(diff)) {
        if (diff < convergence_criteria) {
            result = true;
        }
    } else {
        error_detected_ = true;
    }

    return result;
}

void MF::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        // 欠損箇所だけ計算
        prediction_[index] = user_factors_[missing_data_indices_[index][0]] * item_factors_[missing_data_indices_[index][1]];
    }
}
