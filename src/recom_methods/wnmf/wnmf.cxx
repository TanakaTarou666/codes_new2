#include "wnmf.h"

WNMF::WNMF(int missing_pattern)
    : Recom(missing_pattern),
      user_factors_(),
      item_factors_(),
      prev_user_factors_(),
      prev_item_factors_(),
      transpose_sparse_missing_data_(),
      sparse_prediction_(){
    method_name_ = append_current_time_if_test("WNMF");
}

void WNMF::set_parameters(double latent_dimension_percentage) {
#if defined ARTIFICIALITY
    latent_dimension_ = latent_dimension_percentage;
#else
    if (rs::num_users > rs::num_items) {
        latent_dimension_ = std::round(rs::num_items * latent_dimension_percentage / 100);
    } else {
        latent_dimension_ = std::round(rs::num_users * latent_dimension_percentage / 100);
    }
    if (rs::steps < 50) {
        std::cerr << "MF: \"step\" should be 50 or more.";
        return;
    }
#endif
    parameters_ = {(double)latent_dimension_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
    user_factors_ = Matrix(rs::num_users, latent_dimension_);
    item_factors_ = Matrix(rs::num_items, latent_dimension_);
    return;
}

void WNMF::set_initial_values(int seed) {
    seed *= 1000000;
    std::mt19937_64 mt;
    for (int k_i = 0; k_i < user_factors_.cols(); k_i++) {
        for (int i = 0; i < user_factors_.rows(); i++) {
            mt.seed(seed);
            // ランダムに値生成
            std::uniform_real_distribution<> rand_user_factors_(0.001, 1.0);
            user_factors_(i, k_i) = rand_user_factors_(mt);
            seed++;
        }
        for (int j = 0; j < item_factors_.rows(); j++) {
            mt.seed(seed);
            std::uniform_real_distribution<> rand_item_factors_(0.001, 1.0);
            item_factors_(j, k_i) = rand_item_factors_(mt);
            seed++;
        }
    }

    transpose_sparse_missing_data_ = sparse_missing_data_.transpose();
    sparse_prediction_ = sparse_missing_data_;
}

void WNMF::calculate_factors() {
    prev_item_factors_ = item_factors_;
    prev_user_factors_ = user_factors_;
    // 更新式H
    sparse_prediction_.product(user_factors_, item_factors_);
    Matrix item_numerator = transpose_sparse_missing_data_ * user_factors_;
    Matrix item_denominator = sparse_prediction_.transpose() * user_factors_;
    for (int j = 0; j < rs::num_items; j++) {
        for (int k = 0; k < latent_dimension_; k++) {
            if (item_denominator(j, k) == 0) item_denominator(j, k) = 1.0e-07;
            item_factors_(j, k) *= (item_numerator(j, k) / item_denominator(j, k));
        }
    }
    //  更新式W
    sparse_prediction_.product(user_factors_, item_factors_);
    Matrix user_numerator = sparse_missing_data_ * item_factors_;
    Matrix user_denominator = sparse_prediction_ * item_factors_;
    for (int i = 0; i < rs::num_users; i++) {
        for (int k = 0; k < latent_dimension_; k++) {
            if (user_denominator(i, k) == 0) user_denominator(i, k) = 1.0e-07;
            user_factors_(i, k) *= (user_numerator(i, k) / user_denominator(i, k));
        }
    }
}

double WNMF::calculate_objective_value() {
    double result = 0.0;
    user_factor_values_ = user_factors_.get_values();
    item_factor_values_ = item_factors_.get_values();
    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
            // e = r - p * q
            double err = sparse_missing_data_values_[sparse_missing_data_row_pointers_[i] + j];
            err -= user_factors_[i] * item_factors_[sparse_missing_data_.dense_index(i, j)];
            result += err * err;
        }
    }
    return result;
}

bool WNMF::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = frobenius_norm((prev_user_factors_ - user_factors_)) + frobenius_norm(prev_item_factors_ - item_factors_);
    //std::cout << diff << std::endl;
#else
    objective_value_ = calculate_objective_value();
    double diff = (prev_objective_value_ - objective_value_) / prev_objective_value_;
    prev_objective_value_ = objective_value_;
#endif
    if (std::isfinite(diff)) {
        if (diff < rs::convergence_criteria) {
            result = true;
        }
    } else {
        error_detected_ = true;
    }

    return result;
}

void WNMF::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        // 欠損箇所だけ計算
        prediction_[index] = user_factors_[missing_data_indices_(index, 0)] * item_factors_[missing_data_indices_(index, 1)];
        //  std::cout << "Prediction:" << prediction_[index]
        //              << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_(index, 0), missing_data_indices_(index, 1)) <<
        //              std::endl;
    }
}
