#include "tfcfm_sgd.h"

TFCFMWithSGD::TFCFMWithSGD(int missing_count)
    : FMBase(missing_count), TFCRecom(missing_count), Recom(missing_count), w0_(), prev_w0_(), w_(), prev_w_(), v_(), prev_v_(), e_(), q_(), x_() {
    method_name_ = "TFCFM_ALS";
}

void TFCFMWithSGD::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda,
                                  double reg_parameter, double learning_rate) {
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
    reg_parameter_ = reg_parameter;
    learning_rate_ = learning_rate;
    cluster_size_ = cluster_size;
    fuzzifier_em_ = fuzzifier_em;
    fuzzifier_Lambda_ = fuzzifier_Lambda;
    fuzzifier_Lambda_ = fuzzifier_Lambda;
    parameters_ = {(double)latent_dimension_, (double)cluster_size_, fuzzifier_em_, fuzzifier_Lambda_, reg_parameter_, learning_rate_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void TFCFMWithSGD::set_initial_values(int &seed) {
    w0_ = Vector(cluster_size_, 0.0, "all");
    w_ = Matrix(cluster_size_, rs::num_users + rs::num_items, 0.0);
    v_ = Tensor(cluster_size_, rs::num_users + rs::num_items, latent_dimension_);
    x_ = DSSTensor(sparse_missing_data_, rs::num_users + rs::num_items);
    membership_ = Matrix(cluster_size_, rs::num_users, 1.0 / (double)cluster_size_);
    dissimilarities_ = Matrix(cluster_size_, rs::num_users, 0);

    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int n = 0; n < rs::num_users + rs::num_items; n++) {
            for (int k = 0; k < latent_dimension_; k++) {
                mt.seed(seed);
                // ランダムに値生成
                std::uniform_real_distribution<> rand_v(0.0, 0.001);
                v_[c](n, k) = rand_v(mt);
            }
        }
    }

    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < x_(i, "row"); j++) {
            SparseVector x_element(rs::num_items, 2);
            x_element(0) = 1;
            x_element(0, "index") = i;
            x_element(1) = 1;
            x_element(1, "index") = rs::num_users + x_(i, j, "index");
            x_(i, j) = x_element;
        }
    }

    // データ表示
    // for (int i = 0; i < rs::num_users; i++) {
    //     for (int j = 0; j < x_(i, "row"); j++) {
    //         std::cout << "i:" << i << " j:" << j << " : " << x_(i, j)
    //                   << std::endl;
    //     }
    // }
}

void TFCFMWithSGD::calculate_factors() {
    double sum;
    prev_v_ = v_;
    prev_w_ = w_;
    prev_w0_ = w0_;
    prev_membership_ = membership_;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    double err = (sparse_missing_data_values_[sparse_missing_data_row_pointers_[i] + j] - predict_y(x_(i, j), w0_[c], w_[c], v_[c]));

                    w0_[c] += learning_rate_ * (pow(membership_(c, i), fuzzifier_em_) * err - reg_parameter_ * w0_[c]);
                    for (int a = 0; a < 2; a++) {
                        w_(c, x_(i, j)(a,"index")) += learning_rate_ * (pow(membership_(c, i), fuzzifier_em_) * x_(i, j)(a) * err - reg_parameter_ * w_(c, x_(i, j)(a,"index")));
                        for (int k = 0; k < latent_dimension_; k++) {
                            sum = 0.0;
                            for (int a2 = 0; a2 < 2; a2++) {
                                sum += v_[c](x_(i, j)(a2,"index"), k) * x_(i, j)(a2);
                            }
                            v_[c](x_(i, j)(a,"index"), k) += learning_rate_ * (pow(membership_(c, i), fuzzifier_em_) * err *
                                                                 (x_(i, j)(a) * sum - v_[c](x_(i, j)(a,"index"), k) * x_(i, j)(a) * x_(i, j)(a)) -
                                                             reg_parameter_ * v_[c](x_(i, j)(a,"index"), k));
                        }  // k
                    }      // a
                }
            }  // j
        }      // i
    }          // c

    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    double tmp = 0.0;
                    tmp = (sparse_missing_data_(i, j) - predict_y(x_(i, j), w0_[c], w_[c], v_[c]));
                    dissimilarities_(c, i) = tmp * tmp;
                }
            }
        }
    }
    calculate_membership();
}

double TFCFMWithSGD::calculate_objective_value() {
    double result;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(membership_[c][i], fuzzifier_em_) * dissimilarities_[c][i];
            +1 / (fuzzifier_Lambda_ * (fuzzifier_em_ - 1)) * (pow(membership_[c][i], fuzzifier_em_) - 1);
        }
    }
    return result;
}

bool TFCFMWithSGD::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff =
        squared_norm(prev_w0_ - w0_) + frobenius_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_) + frobenius_norm(prev_membership_ - membership_);
    std::cout << "w0:" << squared_norm(prev_w0_ - w0_) << std::endl;
    std::cout << "w:" << frobenius_norm(prev_w_ - w_) << std::endl;
    std::cout << "v:" << frobenius_norm(prev_v_ - v_) << std::endl;
    std::cout << "m:" << frobenius_norm(prev_membership_ - membership_) << std::endl;
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

void TFCFMWithSGD::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        for (int c = 0; c < cluster_size_; c++) {
            prediction_[index] += membership_[c][missing_data_indices_[index][0]] *
                                  predict_y(x_(missing_data_indices_[index][0], missing_data_indices_[index][1]), w0_[c], w_[c], v_[c]);
        }
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_[index][0], missing_data_indices_[index][1]) << std::endl;
    }
}
