#include "mf_base.h"

MFBase::MFBase(int missing_pattern) : Recom(missing_pattern) {}

void MFBase::set_parameters(double latent_dimension_percentage, double learning_rate, double reg_parameter){
    #if defined ARTIFICIALITY
    latent_dimension_ = latent_dimension_percentage;
#else
    if (rs::num_users > rs::num_items) {
        latent_dimension_ = std::round(rs::num_items * latent_dimension_percentage / 100);
    } else {
        latent_dimension_ = std::round(rs::num_users * latent_dimension_percentage / 100);
    }
#endif
    reg_parameter_ = reg_parameter;
    learning_rate_ = learning_rate;
    parameters_ = {(double)latent_dimension_, reg_parameter_, learning_rate_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
    return;
}

void MFBase::set_initial_values(int seed){
    seed *= 1000000;
    std::mt19937_64 mt;
    for (int k = 0; k < user_factors_.cols(); k++) {
        for (int i = 0; i < user_factors_.rows(); i++) {
            mt.seed(seed);
            // ランダムに値生成
            std::uniform_real_distribution<> rand_user_factors_(0.001, 1.0);
            user_factors_(i, k) = rand_user_factors_(mt);
            seed++;
        }
        for (int j = 0; j < item_factors_.rows(); j++) {
            mt.seed(seed);
            std::uniform_real_distribution<> rand_item_factors_(0.001, 1.0);
            item_factors_(j, k) = rand_item_factors_(mt);
            seed++;
        }
    }
}

double MFBase::calculate_objective_value(){
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
    result += reg_parameter_ * (squared_sum(user_factors_) + squared_sum(item_factors_)) / 2;
    return result;
}

bool MFBase::calculate_convergence_criterion(){
    bool result = false;
#if defined ARTIFICIALITY
    double diff = frobenius_norm((prev_user_factors_ - user_factors_)) + frobenius_norm(prev_item_factors_ - item_factors_);
#else
    objective_value_ = calculate_objective_value();
    double diff = fabs((prev_objective_value_ - objective_value_) / prev_objective_value_);
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

void MFBase::calculate_prediction(){
    for (int index = 0; index < num_missing_value_; index++) {
        // 欠損箇所だけ計算
        prediction_[index] = user_factors_[missing_data_indices_(index, 0)] * item_factors_[missing_data_indices_(index, 1)];
        std::cout << "Prediction:" << prediction_[index]
                  << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_(index, 0), missing_data_indices_(index, 1)) << std::endl;
    }
}
