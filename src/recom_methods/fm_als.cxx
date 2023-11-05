#include "fm_als.h"

FMWithALS::FMWithALS(int missing_count)
    : FMBase(missing_count),
      Recom(missing_count),
      w0_(),
      prev_w0_(),
      w_(),
      prev_w_(),
      v_(),
      prev_v_(),
      e_(),
      q_(),
      x_() {
    method_name_ = "FM_ALS";
}

void FMWithALS::set_parameters(double latent_dimension_percentage,
                               double reg_parameter) {
#if defined ARTIFICIALITY
    latent_dimension_ = latent_dimension_percentage;
#else
    if (num_users > num_items) {
        latent_dimension_ =
            std::round(num_items * latent_dimension_percentage / 100);
    } else {
        latent_dimension_ =
            std::round(num_users * latent_dimension_percentage / 100);
    }
    if (steps < 50) {
        std::cerr << "MF: \"step\" should be 50 or more.";
        return;
    }
#endif
    reg_parameter_ = reg_parameter;
    parameters_ = {(double)latent_dimension_, reg_parameter_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void FMWithALS::train() {
    int error_count = 0;
    double best_objective_value = DBL_MAX;
    for (int initial_value_index = 0; initial_value_index < num_initial_values;
         initial_value_index++) {
        std::cout << method_name_ << ": initial setting " << initial_value_index
                  << std::endl;
        set_initial_values(initial_value_index);
        error_detected_ = false;
#ifndef ARTIFICIALITY
        prev_objective_value_ = DBL_MAX;
#endif
        precompute();
        for (int step = 0; step < steps; step++) {
            std::cout << "step:" << step << "\t"
                      << "objective_value:" << calculate_objective_value()
                      << "\t";
            calculate_Wo_w_v();
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

void FMWithALS::set_initial_values(int &seed) {
    w0_ = 0.0;
    w_ = Vector(num_users + num_items, 0.0, "all");
    v_ = Matrix(num_users + num_items, latent_dimension_);
    e_ = Vector(sparse_missing_data_.nnz() - num_missing_value_, 0.0, "all");
    q_ = Matrix(sparse_missing_data_.nnz() - num_missing_value_,
                latent_dimension_);
    x_ = DSSTensor(sparse_missing_data_, num_users + num_items);

    std::mt19937_64 mt;
    for (int n = 0; n < num_users + num_items; n++) {
        for (int k = 0; k < latent_dimension_; k++) {
            mt.seed(seed);
            // ランダムに値生成
            std::uniform_real_distribution<> rand_v(0.0, 0.001);
            v_(n, k) = rand_v(mt);
            v_(n, k) = 0.01 * k + 0.01;
            seed++;
        }
    }

    for (int i = 0; i < num_users; i++) {
        for (int j = 0; j < x_(i, "row"); j++) {
            SparseVector x_element(num_items, 2);
            x_element(0) = 1;
            x_element(0, "index") = i;
            x_element(1) = 1;
            x_element(1, "index") = num_users + x_(i, j, "index");
            x_(i, j) = x_element;
        }
    }

    // データ表示
    // for (int i = 0; i < num_users; i++) {
    //     for (int j = 0; j < x_(i, "row"); j++) {
    //         std::cout << "i:" << i << " j:" << j << " : " << x_(i, j)
    //                   << std::endl;
    //     }
    // }
}

void FMWithALS::precompute() {
    int l = 0;
    for (int i = 0; i < sparse_missing_data_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                e_[l] = predict_y(x_(i, j), w0_, w_, v_) -
                        sparse_missing_data_(i, j);
                for (int f = 0; f < latent_dimension_; ++f) {
                    q_(l, f) = 0.0;
                    for (int j_ = 0; j_ < x_(i, j).nnz(); ++j_) {
                        q_(l, f) += x_(i, j)(j_) * v_(x_(i, j)(j_, "index"), f);
                    }
                }
                l++;
            }
        }
    }
    std::cout << q_[80] << std::endl;
}

void FMWithALS::calculate_Wo_w_v() {
    prev_v_ = v_;
    prev_w_ = w_;
    prev_w0_ = w0_;
    std::cout << "w0:" << w0_ << "\t";
    std::cout << "w:" << w_[80] << "\t";
    std::cout << "v:" << v_[0] << "\t";
    double numerator_w0 = 0;
    double denominator_w0 = 0;
    int l = 0;
    for (int i = 0; i < sparse_missing_data_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                numerator_w0 += (e_[l] - w0_);
                l++;
            }
        }
    }
    double w0a = -numerator_w0 / l;
    for (l = 0; l < e_.size(); l++) e_[l] += (w0a - w0_);
    w0_ = w0a;
    std::cout << "e" << e_[0] << ", l:" << 0 << std::endl;

    // 1-way interactions
    // double numerator_w[w_.size()] = {};
    // double denominator_w[w_.size()] = {};
    // l = 0;
    // for (int i = 0; i < sparse_missing_data_.rows(); i++) {
    //     for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
    //         if (sparse_missing_data_(i, j) != 0) {
    //             for (int a = 0; a < x_(i, j).nnz(); ++a) {
    //                 numerator_w[x_(i, j)(a, "index")] +=
    //                     (e_[l] - w_[x_(i, j)(a, "index")] * x_(i, j)(a)) *
    //                     x_(i, j)(a);
    //                 denominator_w[x_(i, j)(a, "index")] +=
    //                     x_(i, j)(a) * x_(i, j)(a);
    //                     //if(x_(i, j)(a, "index")==80) std::cout<< e_[l]<<",
    //                     l:"<<l <<std::endl;
    //             }
    //             l++;
    //         }
    //     }
    // }
    for (int a = 0; a < 2; ++a) {
        double numerator_w = 0.0;
        double denominator_w = 0.0;
        l=0.0;
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    numerator_w +=
                        (e_[l] - w_[x_(i, j)(a, "index")] * x_(i, j)(a)) *
                        x_(i, j)(a);
                    denominator_w += x_(i, j)(a) * x_(i, j)(a);
                    l++;
                }
            }
            double wa = -numerator_w / denominator_w;
        }
        
    }

    double wa[w_.size()] = {};
    for (int a = 0; a < w_.size(); ++a) {
        wa[a] = -numerator_w[a] / denominator_w[a];
    }
    l = 0;
    for (int i = 0; i < sparse_missing_data_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                for (int a = 0; a < x_(i, j).nnz(); ++a) {
                    e_[l] +=
                        (wa[x_(i, j)(a, "index")] - w_[x_(i, j)(a, "index")]) *
                        x_(i, j)(a);
                }
                l++;
            }
        }
    }
    for (int a = 0; a < w_.size(); ++a) {
        w_[a] = wa[a];
    }

    // 2-way interactions
    for (int f = 0; f < latent_dimension_; ++f) {
        double h_value[e_.size()] = {};
        int l = 0;
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    for (int a = 0; a < x_(i, j).nnz(); ++a) {
                        h_value[l] =
                            -x_(i, j)(a) *
                            (x_(i, j)(a) * v_(x_(i, j)(a, "index"), f) -
                             q_(l, f));
                    }
                    l++;
                }
            }
        }
        double numerator_v[v_.rows()] = {};
        double denominator_v[v_.rows()] = {};
        l = 0;
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    for (int a = 0; a < x_(i, j).nnz(); ++a) {
                        numerator_v[x_(i, j)(a, "index")] +=
                            (e_[l] - v_(x_(i, j)(a, "index"), f) * h_value[l]) *
                            h_value[l];
                        denominator_v[x_(i, j)(a, "index")] +=
                            h_value[l] * h_value[l];
                    }
                    l++;
                }
            }
        }

        double va[v_.rows()] = {};
        for (int a = 0; a < v_.rows(); ++a) {
            va[a] = -numerator_v[a] / denominator_v[a];
        }
        l = 0;
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    for (int a = 0; a < x_(i, j).nnz(); ++a) {
                        e_[l] += (va[x_(i, j)(a, "index")] -
                                  v_(x_(i, j)(a, "index"), f)) *
                                 h_value[l];
                        q_(l, f) += (va[x_(i, j)(a, "index")] -
                                     v_(x_(i, j)(a, "index"), f)) *
                                    x_(i, j)(a);
                    }
                    l++;
                }
            }
        }
        for (int a = 0; a < v_.rows(); ++a) {
            v_(a, f) = va[a];
        }
    }
}

double FMWithALS::calculate_objective_value() {
    double result = 0.0;
    for (int i = 0; i < sparse_missing_data_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                double tmp = 0.0;
                tmp = (sparse_missing_data_(i, j) -
                       predict_y(x_(i, j), w0_, w_, v_));
                result += tmp * tmp;
            }
        }
    }
    return result;
}

bool FMWithALS::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = abs(prev_w0_ - w0_) + squared_norm(prev_w_ - w_) +
                  frobenius_norm(prev_v_ - v_);
    std::cout << "diff:" << diff << "\t" << std::endl;
    // std::cout << "w0:" << w0_ << std::endl;
    // std::cout << "w:" << w_ << std::endl;
    // std::cout << "v:" << v_ << std::endl;
#else
    objective_value_ = calculate_objective_value();
    double diff =
        (prev_objective_value_ - objective_value_) / prev_objective_value_;
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

void FMWithALS::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        prediction_[index] += predict_y(x_(missing_data_indices_[index][0],
                                           missing_data_indices_[index][1]),
                                        w0_, w_, v_);
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:"
        //           << sparse_correct_data_(missing_data_indices_[index][0],
        //                                   missing_data_indices_[index][1])
        //           << std::endl;
    }
}