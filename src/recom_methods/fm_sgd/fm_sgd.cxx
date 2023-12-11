#include "fm_sgd.h"

FMWithSGD::FMWithSGD(int missing_count) : FMBase(missing_count), Recom(missing_count), w_(), prev_w_(), v_(), prev_v_(), x_() {
    method_name_ = append_current_time_if_test("FM_SGD");
}

void FMWithSGD::set_parameters(double latent_dimension_percentage, double reg_parameter, double learning_rate) {
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
    parameters_ = {(double)latent_dimension_, reg_parameter_, learning_rate_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void FMWithSGD::set_initial_values(int seed) {
    seed *= 1000000;
    w0_ = 0.0;
    w_ = Vector(rs::num_users + rs::num_items, 0.0, "all");
    v_ = Matrix(latent_dimension_,rs::num_users + rs::num_items);
    x_ = DSSTensor(sparse_missing_data_, rs::num_users + rs::num_items);

    std::mt19937_64 mt;
    for (int n = 0; n < rs::num_users + rs::num_items; n++) {
        for (int k = 0; k < latent_dimension_; k++) {
            mt.seed(seed);
            // ランダムに値生成
            std::uniform_real_distribution<> rand_v(-0.01, 0.01);
            v_(k,n) = rand_v(mt);
            // v_(n, k) = 0.01 * k + 0.01 * n;
            seed++;
        }
    }

    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < x_(i, "row"); j++) {
            SparseVector x_element(rs::num_items, 2);
            x_element(0) = 1;
            x_element.dense_index(0) = i;
            x_element(1) = 1;
            x_element.dense_index(1) = rs::num_users + x_.dense_index(i, j);
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

void FMWithSGD::calculate_factors() {
    double sum;
    prev_v_ = v_;
    prev_w_ = w_;
    prev_w0_ = w0_;

    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                SparseVector tmp_x = x_(i, j);
                // 予測値(prediction)の計算
                //  線形項の計算
                double prediction = w0_;
                double linearTerm = 0.0;
                for (int a = 0; a < 2; a++) {
                    linearTerm += w_[tmp_x.dense_index(a)] * x_(i, j)(a);
                }
                // 交互作用項の計算
                double sum[latent_dimension_] = {};
                double squareSum[latent_dimension_] = {};
                for (int factor = 0; factor < latent_dimension_; factor++) {
                    for (int a = 0; a < 2; a++) {
                        double tmp = v_(factor,tmp_x.dense_index(a)) * tmp_x(a);
                        sum[factor] += tmp;
                        squareSum[factor] += tmp * tmp;
                    }
                    prediction += 0.5 * (sum[factor] * sum[factor] - squareSum[factor]);
                }
                prediction += linearTerm;

                // 予測値との差の計算
                double err = (sparse_missing_data_(i, j) - prediction);

                // w0の更新
                w0_ += learning_rate_ * (2 * err - reg_parameter_ * w0_);
                // wの更新
                for (int a = 0; a < 2; a++) {
                    w_[tmp_x.dense_index(a)] += learning_rate_ * (2 * tmp_x(a) * err - reg_parameter_ * w_[tmp_x.dense_index(a)]);
                }
                // vの更新
                for (int a = 0; a < 2; a++) {
                    double tmp_x_value = tmp_x(a);
                    double tmp_x_index = tmp_x.dense_index(a);
                    for (int factor = 0; factor < latent_dimension_; factor++) {
                        v_(factor,tmp_x_index) += learning_rate_ * (2 * err * tmp_x_value * (sum[factor] - v_(factor,tmp_x_index) * tmp_x_value) -
                                                                     reg_parameter_ * v_(factor,tmp_x_index));
                    }
                }
            }
        }  // j
    }      // i
}

double FMWithSGD::calculate_objective_value() {
    double result = 0.0;
    for (int i = 0; i < sparse_missing_data_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                double tmp = 0.0;
                tmp = (sparse_missing_data_(i, j) - predict_y(x_(i, j), w0_, w_, v_));
                result += tmp * tmp;
            }
        }
    }
    result += reg_parameter_ * (fabs(w0_) + squared_sum(w_) + squared_sum(v_));
    return result;
}

bool FMWithSGD::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = abs(prev_w0_ - w0_) + squared_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_);
    // std::cout << "L:" << calculate_objective_value() << "\t";
    // std::cout << "diff:" << diff << "\t" << std::endl;
    // std::cout << "w0:" << abs(prev_w0_ - w0_) << std::endl;
    // std::cout << "w:" << squared_norm(prev_w_ - w_) << std::endl;
    // std::cout << "v:" << frobenius_norm(prev_v_ - v_) << std::endl;
#else
    objective_value_ = calculate_objective_value();
    double diff = (prev_objective_value_ - objective_value_) / prev_objective_value_;
    prev_objective_value_ = objective_value_;
        std::cout << "L:" << calculate_objective_value() << "\t";
    std::cout << "diff:" << diff << "\t" << std::endl;
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

void FMWithSGD::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        // std::cout << "m:"<< missing_data_indices_(index, 0)<<":"<<sparse_missing_data_cols_[index] << std::endl;
        prediction_[index] = predict_y(x_(missing_data_indices_(index, 0), sparse_missing_data_cols_[index]), w0_, w_, v_);
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:"
        //           << sparse_correct_data_(missing_data_indices_(index,0),
        //                                   sparse_missing_data_cols_[index])
        //           << std::endl;
    }
    std::cout << "end"<< "\t" << std::endl;
}
