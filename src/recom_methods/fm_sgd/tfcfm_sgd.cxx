#include "tfcfm_sgd.h"

TFCFMWithSGD::TFCFMWithSGD(int missing_count)
    : FMBase(missing_count), TFCRecom(missing_count), Recom(missing_count), w0_(), prev_w0_(), w_(), prev_w_(), v_(), prev_v_(), x_() {
    method_name_ = append_current_time_if_test("TFCFM_SGD");
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
    fuzzifier_lambda_ = fuzzifier_Lambda;
    fuzzifier_lambda_ = fuzzifier_Lambda;
    parameters_ = {(double)latent_dimension_, (double)cluster_size_, fuzzifier_em_, fuzzifier_lambda_, reg_parameter_, learning_rate_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void TFCFMWithSGD::set_initial_values(int seed) {
    seed *= 1000000;
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
                std::uniform_real_distribution<> rand_v(-0.01, 0.01);
                v_[c](n, k) = rand_v(mt);
                seed++;
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
        Matrix tmp_v = v_[c];
        for (int i = 0; i < rs::num_users; i++) {
            double tmp_membership = pow(membership_(c, i), fuzzifier_em_);
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    SparseVector tmp_x = x_(i, j);
                    //  線形項の計算
                    double prediction = w0_[c];
                    double linearTerm = 0.0;
                    for (int a = 0; a < 2; a++) {
                        linearTerm += w_(c, tmp_x(a, "index")) * x_(i, j)(a);
                    }
                    // 交互作用項の計算
                    double sum[latent_dimension_] = {};
                    double squareSum[latent_dimension_] = {};
                    for (int factor = 0; factor < latent_dimension_; factor++) {
                        for (int a = 0; a < 2; a++) {
                            double tmp = tmp_v(tmp_x(a, "index"), factor) * tmp_x(a);
                            sum[factor] += tmp;
                            squareSum[factor] += tmp * tmp;
                        }
                        prediction += 0.5 * (sum[factor] * sum[factor] - squareSum[factor]);
                    }
                    prediction += linearTerm;

                    // 予測値との差の計算
                    double err = (sparse_missing_data_(i, j) - prediction);

                    // w0の更新
                    w0_[c] += learning_rate_ * (2 * tmp_membership * err - reg_parameter_ * w0_[c]);
                    // wの更新
                    for (int a = 0; a < 2; a++) {
                        w_(c, tmp_x(a, "index")) +=
                            learning_rate_ * (2 * tmp_x(a) * tmp_membership * err - reg_parameter_ * w_(c, tmp_x(a, "index")));
                    }
                    
                    // vの更新
                    for (int a = 0; a < 2; a++) {
                        double tmp_x_value = tmp_x(a);
                        double tmp_x_index = tmp_x(a, "index");
                        for (int factor = 0; factor < latent_dimension_; factor++) {
                            tmp_v(tmp_x_index, factor) +=
                                learning_rate_ * (2 * tmp_membership * err * tmp_x_value * (sum[factor] - tmp_v(tmp_x_index, factor) * tmp_x_value) -
                                                  reg_parameter_ * tmp_v(tmp_x_index, factor));
                        }
                    }
                }
            }  // j
        }      // i
        v_[c] = tmp_v;
    }  // c

    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    double tmp = (sparse_missing_data_(i, j) - predict_y(x_(i, j), w0_[c], w_[c], v_[c]));
                    dissimilarities_(c, i) += tmp * tmp;
                }
            }
        }
    }
    calculate_membership();
}

double TFCFMWithSGD::calculate_objective_value() {
    double result = 0;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += dissimilarities_(c, i) * pow(membership_(c, i), fuzzifier_em_) +
                      1 / (fuzzifier_lambda_ * (fuzzifier_em_ - 1)) * (pow(membership_(c, i), fuzzifier_em_) - membership_(c, i));
        }
    }
    result += reg_parameter_ * (squared_norm(w0_) + frobenius_norm(w_) + frobenius_norm(v_));
    return result;
}

bool TFCFMWithSGD::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff =
        squared_norm(prev_w0_ - w0_) + frobenius_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_) + frobenius_norm(prev_membership_ - membership_);
    std::cout << " diff:" << diff << " L:" << calculate_objective_value() << "\t";
    std::cout << "w0:" << squared_norm(prev_w0_ - w0_) << "\t";
    std::cout << "w:" << frobenius_norm(prev_w_ - w_) << "\t";
    std::cout << "v:" << frobenius_norm(prev_v_ - v_) << "\t";
    std::cout << "m:" << frobenius_norm(prev_membership_ - membership_) << "\t";
    std::cout << std::endl;
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
            prediction_[index] += membership_(c, missing_data_indices_(index, 0)) *
                                  predict_y(x_(missing_data_indices_(index, 0), missing_data_indices_(index, 1)), w0_[c], w_[c], v_[c]);
        }
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_[index][0], missing_data_indices_[index][1]) << std::endl;
    }
}
