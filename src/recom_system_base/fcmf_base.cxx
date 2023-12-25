#include "fcmf_base.h"

FCMFBase::FCMFBase(int missing_pattern) : Recom(missing_pattern), membership_(), prev_membership_(), dissimilarities_() {}

void FCMFBase::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_lambda,
                              double reg_parameter, double learning_rate) {
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
    cluster_size_ = cluster_size;
    fuzzifier_em_ = fuzzifier_em;
    fuzzifier_lambda_ = fuzzifier_lambda;
    parameters_ = {(double)latent_dimension_, reg_parameter_, learning_rate_, (double)cluster_size_, fuzzifier_em_, fuzzifier_lambda_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void FCMFBase::set_initial_values(int seed) {
    seed *= 1000000;
    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int k_i = 0; k_i < user_factors_.cols(); k_i++) {
            for (int i = 0; i < user_factors_.rows(); i++) {
                mt.seed(seed);
                // ランダムに値生成
                std::uniform_real_distribution<> rand_user_factors_(0.001, 1.0);
                user_factors_[c](i, k_i) = rand_user_factors_(mt);
                seed++;
            }
            for (int j = 0; j < item_factors_.rows(); j++) {
                mt.seed(seed);
                std::uniform_real_distribution<> rand_item_factors_(0.001, 1.0);
                item_factors_[c](j, k_i) = rand_item_factors_(mt);
                seed++;
            }
        }
    }
    membership_ = Matrix(cluster_size_, rs::num_users, 1.0 / (double)cluster_size_);
    dissimilarities_ = Matrix(cluster_size_, rs::num_users, 0);
    for (int k = 0; k < rs::num_users; k++) {
        double tmp_Mem[cluster_size_];
        tmp_Mem[cluster_size_ - 1] = 1.0;
        for (int i = 0; i < cluster_size_ - 1; i++) {
            mt.seed(seed);
            std::uniform_real_distribution<> rand_p(0.01, 1.0 / (double)cluster_size_);
            tmp_Mem[i] = rand_p(mt);
            tmp_Mem[cluster_size_ - 1] -= tmp_Mem[i];
            seed++;
        }
        // [0, 99] 範囲の一様乱数
        for (int i = 0; i < cluster_size_; i++) {
            mt.seed(seed);
            std::uniform_int_distribution<> rand100(0, 99);
            int r = rand100(mt) % (1 + i);
            double tmp = tmp_Mem[i];
            tmp_Mem[i] = tmp_Mem[r];
            tmp_Mem[r] = tmp;
            seed++;
        }
        for (int i = 0; i < cluster_size_; i++) {
            membership_(i, k) = tmp_Mem[i];
        }
    }
}

double FCMFBase::calculate_objective_value() {
    double result = 0;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(membership_(c, i), fuzzifier_em_) * dissimilarities_(c, i) +
                      1 / (fuzzifier_lambda_ * (fuzzifier_em_ - 1)) * (pow(membership_(c, i), fuzzifier_em_) - membership_(c, i));
        }
    }
    result += reg_parameter_ * (squared_sum(user_factors_) + squared_sum(item_factors_))/2;
    return result;
}

bool FCMFBase::calculate_convergence_criterion() {
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

void FCMFBase::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        for (int c = 0; c < cluster_size_; c++) {
            prediction_[index] += membership_(c, missing_data_indices_(index, 0)) *
                                  (user_factors_[c][missing_data_indices_(index, 0)] * item_factors_[c][missing_data_indices_(index, 1)]);
        }
        //  std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_(index,0),missing_data_indices_[index][1]) << std::endl;
    }
}
