#include "recom.h"

Recom::Recom(int num_missing_value)
    : num_missing_value_(num_missing_value),
      sparse_correct_data_(rs::num_users, rs::num_items),
      sparse_missing_data_(rs::num_users, rs::num_items),

      missing_data_indices_(num_missing_value, 2),
      sparse_missing_data_cols_(num_missing_value),
      prediction_(num_missing_value, 0, "all"),
      mae_(rs::missing_pattern),
      auc_(rs::missing_pattern),
      tp_fn_((int)rs::max_value * 100, 0.0, "all"),
      fp_tn_((int)rs::max_value * 100, 0.0, "all") {}

void Recom::input(std::string file_name) {
    std::ifstream ifs(file_name);
    if (!ifs) {
        std::cerr << "directory_name:" << file_name << ": could not open." << std::endl;
        exit(1);
    }

    int *row_pointers = new int[rs::num_users + 1]();
    int row_sizes[rs::num_users];
    std::vector<std::vector<double>> correct_data;
    std::vector<std::vector<int>> correct_data_index;
    row_pointers[0] = 0;
    for (int cnt = 0; cnt < rs::num_users; cnt++) {
        std::vector<int> correct_data_index_row;
        std::vector<double> correct_data_row;
        ifs >> row_sizes[cnt];
        int col_index;
        double data;
        row_pointers[cnt + 1] = row_pointers[cnt] + row_sizes[cnt];
        for (int ell = 0; ell < row_sizes[cnt]; ell++) {
            ifs >> col_index >> data;
            correct_data_index_row.push_back(col_index);
            correct_data_row.push_back(data);
        }
        correct_data_index.push_back(correct_data_index_row);
        correct_data.push_back(correct_data_row);
    }
    ifs.close();
    sparse_correct_data_.set_row_pointers(row_pointers);

    int *col_indices = new int[row_pointers[rs::num_users]]();
    double *values = new double[row_pointers[rs::num_users]]();
    int i = 0;
    for (int cnt = 0; cnt < rs::num_users; cnt++) {
        for (int ell = 0; ell < row_sizes[cnt]; ell++) {
            col_indices[i] = correct_data_index[cnt][ell];
            values[i] = correct_data[cnt][ell];
            i++;
        }
    }

    sparse_correct_data_.set_col_indices(col_indices);
    sparse_correct_data_.set_values(values);
    sparse_correct_data_.set_nnz(row_pointers[rs::num_users]);
}

void Recom::revise_missing_values(void) {
    SparseMatrix tmp_sparse_missing_data = sparse_correct_data_;

    int tmprow, tmpcol;
    for (int m = 0; m < num_missing_value_;) {
        /****乱数生成****/
        std::mt19937_64 mt;
        mt.seed(seed_);
        std::uniform_int_distribution<> randRow(0, rs::num_users - 1);
        // ランダムに行番号生成
        tmprow = randRow(mt);
        int col_size = sparse_correct_data_.get_row_pointers()[tmprow + 1] - sparse_correct_data_.get_row_pointers()[tmprow];
        std::uniform_int_distribution<> randCol(0, col_size - 1);
        // ランダムに列番号生成
        tmpcol = randCol(mt);
        // データ行すべて欠損させないように,一行5要素は残す
        int c = 0;
        for (int i = 0; i < col_size; i++)
            if (tmp_sparse_missing_data(tmprow, i) == 0) c++;
        // 既に欠損していない場合
        if (tmp_sparse_missing_data(tmprow, tmpcol) > 0 && col_size - c > 5) {
            // 要素を0にする
            tmp_sparse_missing_data(tmprow, tmpcol) = 0;
            // 欠損した行番号を保存
            missing_data_indices_(m, 0) = tmprow;
            // 欠損した列番号を保存
            missing_data_indices_(m, 1) = tmp_sparse_missing_data.dense_index(tmprow, tmpcol);
            // スパースデータの列番号を保存
            sparse_missing_data_cols_[m] = tmpcol;
            m++;
        }
        seed_++;
    }
    sparse_missing_data_ = tmp_sparse_missing_data.remove_zeros();
    sparse_missing_data_values_ = sparse_missing_data_.get_values();
    sparse_missing_data_row_pointers_ = sparse_missing_data_.get_row_pointers();
    sparse_missing_data_col_indices_ = sparse_missing_data_.get_col_indices();
    for (int i = 0; i < rs::num_users; i++) {
        sparse_missing_data_row_nnzs_[i] = sparse_missing_data_row_pointers_[i+1] - sparse_missing_data_row_pointers_[i];
    }
    missing_num_samples_ = rs::num_samples - num_missing_value_;
    return;
}

void Recom::train() {
    int error_count = 0;
    double best_objective_value = DBL_MAX;
    for (int initial_value_index = 0; initial_value_index < rs::num_initial_values; initial_value_index++) {
        std::cout << method_name_ << ": initial setting " << initial_value_index << std::endl;
        set_initial_values(initial_value_index);
        error_detected_ = false;
#ifndef ARTIFICIALITY
        prev_objective_value_ = DBL_MAX;
#endif
        std::cout << method_name_ << ": train start " << std::endl;
        for (int step = 0; step < rs::steps; step++) {
            calculate_factors();
            // 収束条件
            std::cout << ": step: " << step << "\t" <<"L:"<<calculate_objective_value()<<std::endl;;
            if (calculate_convergence_criterion()) {
                 std::cout << ": step: " << step << std::endl;
                break;
            }
            //std::cout << std::endl;
            if (step == rs::steps - 1) {
                // error_detected_ = true;
                std::cout << ": step: " << step << " error" << std::endl;
                break;
            }
            if (error_detected_) {
                std::cout << ": step: " << step << " error" << std::endl;
                break;
            }
        }

        if (error_detected_) {
            error_count++;
            // 初期値全部{NaN出た or step上限回更新して収束しなかった} => 1を返して終了
            if (error_count == rs::num_initial_values) {
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
    return;
}

void Recom::set_initial_values(int seed) {}

void Recom::calculate_factors() { return; }

double Recom::calculate_objective_value() { return 0; }

bool Recom::calculate_convergence_criterion() { return false; }

void Recom::calculate_prediction() { return; }

void Recom::calculate_mae(int current_missing_pattern) {
    double result = 0.0;
    for (int m = 0; m < num_missing_value_; m++) {
        result += fabs(sparse_correct_data_(missing_data_indices_(m, 0), sparse_missing_data_cols_[m]) - prediction_[m]);
    }

    mae_[current_missing_pattern] = result / (double)num_missing_value_;

    std::ofstream ofs(dirs_[0] + "/" + method_name_ + "MAE.txt", std::ios::app);
    ofs << num_missing_value_ << "\t" << seed_ << "\t" << current_missing_pattern << "\t" << std::fixed << std::setprecision(10)
        << result / (double)num_missing_value_ << std::endl;
    ofs.close();
    return;
}

void Recom::calculate_roc(int current_missing_pattern) {
    for (int index = 1; index < (int)rs::max_value * 100; index++) {
        double TP = 0.0, FP = 0.0, FN = 0.0, TN = 0.0;
        // 閾値の設定
        double siki = (double)index / 100.0;
        for (int m = 0; m < num_missing_value_; m++) {
            // 正解値が閾値以上かつ，予測値が閾値以上場合
            if ((siki <= sparse_correct_data_(missing_data_indices_(m, 0), sparse_missing_data_cols_[m])) && (siki <= prediction_[m])) TP += 1.0;
            // 正解値が閾値を下回ったかつ，予測値が閾値上回った場合
            else if ((siki > sparse_correct_data_(missing_data_indices_(m, 0), sparse_missing_data_cols_[m])) && (siki <= prediction_[m]))
                FP += 1.0;
            // 正解値が閾値上回ったかつ，予測値が閾値を下回った場合
            else if ((siki <= sparse_correct_data_(missing_data_indices_(m, 0), sparse_missing_data_cols_[m])) && (siki > prediction_[m]))
                FN += 1.0;
            // それ以外
            else if ((siki > sparse_correct_data_(missing_data_indices_(m, 0), sparse_missing_data_cols_[m])) && (siki > prediction_[m]))
                TN += 1.0;
            else
                continue;
        }
        if (TP + TN == num_missing_value_) {
            tp_fn_[index] = 1.0;
            fp_tn_[index] = 1.0;
        }
        // Recall，Falloutの計算
        else {
            tp_fn_[index] = TP / (TP + FN);
            fp_tn_[index] = FP / (FP + TN);
            if ((TP + FN) == 0 || (FP + TN) == 0) {
                tp_fn_[index] = 1.0;
                fp_tn_[index] = 1.0;
            }
        }
    }
    std::string ROC_STR = dirs_[0] + "/ROC/choice/" + method_name_ + "ROC" + std::to_string(num_missing_value_) + "_" +
                          std::to_string(current_missing_pattern) + "sort.txt";
    // ROCでプロットする点の数
    int max_index = (int)rs::max_value * 100;
    // 一旦保存
    Vector False = fp_tn_;
    Vector True = tp_fn_;
    std::ofstream ofs(ROC_STR, std::ios::app);
    if (!ofs)
        std::cerr << "ファイルオープン失敗(Recom::roc)\n";
    else {
        // 横軸でソート
        sort(False, True, max_index);
        for (int i = 0; i < max_index; i++) ofs << std::fixed << std::setprecision(10) << False[i] << "\t" << True[i] << std::endl;
    }
    ofs.close();
    return;
}

void Recom::sort(Vector &fal, Vector &tru, int index) {
    double tmp1, tmp2;
    for (int j = 0; j < index - 1; j++) {
        if (fal[j] == 1 && tru[j] != 1) {
            std::cout << "error: TPR != 1 (FPR = 1)" << std::endl;
        }
        for (int k = j + 1; k < index; k++) {
            if (fal[j] > fal[k]) {
                tmp1 = fal[j];
                tmp2 = tru[j];
                fal[j] = fal[k];
                tru[j] = tru[k];
                fal[k] = tmp1;
                tru[k] = tmp2;
            }
        }
    }
    for (int j = 0; j < index - 1; j++) {
        for (int k = j + 1; k < index; k++) {
            if (fal[j] == fal[k] && tru[j] > tru[k]) {
                tmp1 = fal[j];
                tmp2 = tru[j];
                fal[j] = fal[k];
                tru[j] = tru[k];
                fal[k] = tmp1;
                tru[k] = tmp2;
            }
        }
    }

    return;
}

void Recom::precision_summury() {
    int max = (int)rs::max_value * 100;
    for (int method = 0; method < (int)dirs_.size(); method++) {
        double rocarea = 0.0;
        for (int x = 0; x < rs::missing_pattern; x++) {
            auc_[x] = 0.0;
            Vector array1(max, 0.0, "all"), array2(max, 0.0, "all");
            std::ifstream ifs(dirs_[method] + "/ROC/choice/" + method_name_ + "ROC" + std::to_string(num_missing_value_) + "_" + std::to_string(x) +
                              "sort.txt");
            if (!ifs) {
                std::cerr << "ファイルinput失敗(precision_summury):trials:" << x << "miss:" << num_missing_value_ << std::endl;
                std::cout << dirs_[method] + "/ROC/choice/" + method_name_ + "ROC" + std::to_string(num_missing_value_) + "_" + std::to_string(x) +
                                 "sort.txt"
                          << std::endl;
                break;
            }
            for (int i = 0; i < max; i++) ifs >> array1[i] >> array2[i];
            ifs.close();
            for (int i = 0; i < max - 1; i++) {
                if ((array1[i] < array1[i + 1])) {
                    double low = array1[i + 1] - array1[i];
                    double height = fabs(array2[i + 1] - array2[i]);
                    double squarearea;
                    if (array2[i] < array2[i + 1]) {
                        squarearea = low * array2[i];
                    } else {
                        squarearea = low * array2[i + 1];
                    }
                    double triangle = (low * height) / 2.0;
                    rocarea += squarearea + triangle;
                    auc_[x] += squarearea + triangle;
                }
            }
        }

        double sumMAE = 0.0, sumF = 0.0;
        std::ofstream ofs(dirs_[method] + "/ChoicedMaeAuc.txt", std::ios::app);
        if (!ofs) {
            std::cerr << "ファイルopen失敗: choice_mae_f\n";
            exit(1);
        }
        for (int i = 0; i < rs::missing_pattern; i++) {
            ofs << std::fixed << std::setprecision(10) << mae_[i] << "\t" << auc_[i] << std::endl;
        }
    }
    seed_ = 0;
    return;
}

void Recom::tally_result() {
    std::string result = std::filesystem::current_path();
    result = result + "/../../RESULT";
    // 出力先のディレクトリ作成
    std::string directory_name_of_auc = result + "/AUC";
    std::string directory_name_of_mae = result + "/MAE";
    mkdir(directory_name_of_auc.c_str(), 0755);
    mkdir(directory_name_of_mae.c_str(), 0755);
    directory_name_of_auc += "/summay";
    directory_name_of_mae += "/summay";
    mkdir(directory_name_of_auc.c_str(), 0755);
    mkdir(directory_name_of_mae.c_str(), 0755);

    // 出力のファイル名
    std::string output_file_name_of_auc = method_name_ + "_" + rs::data_name + std::to_string(num_missing_value_) + "_summary" + ".txt";
    std::string output_file_name_of_mae = method_name_ + "_" + rs::data_name + std::to_string(num_missing_value_) + "_summary" + ".txt";

    // 入力のファイル名
    std::string input_file_name = result + "/" + method_name_ + "/" + method_name_ + "_" + rs::data_name + std::to_string(num_missing_value_);
    std::string tmp_parameters = "/";
    for (int i = 0; i < (int)parameters_.size(); i++) {
        std::ostringstream oss;
        oss << std::setprecision(10) << parameters_[i];
        std::string p(oss.str());
        tmp_parameters += p;
        if (i < (int)parameters_.size() - 1) tmp_parameters += "_";
    }
    tmp_parameters += "/ChoicedMaeAuc.txt";
    input_file_name += tmp_parameters;

    // ファイル出力
    std::ifstream ifs(input_file_name);
    if (!ifs) {
        std::cerr << "ファイルopen失敗: " << input_file_name << std::endl;
    } else {
        std::cerr << "ファイルopen成功: " << input_file_name << std::endl;
        double tmp_auc[rs::missing_pattern];
        double tmp_mae[rs::missing_pattern];
        int i = 0;
        for (std::string str; std::getline(ifs, str); i++) {
            if (i >= rs::missing_pattern) {
                std::cout << "tmp_aucの容量を超えました。" << std::endl;
                break;
            }
            std::istringstream stream(str);
            int j = 0;
            std::string tmp;
            for (std::string tmp; getline(stream, tmp, '\t'); j++) {
                if (j == 1) tmp_auc[i] = std::stod(tmp);
                if (j == 0) tmp_mae[i] = std::stod(tmp);
            }
        }
        // tmp_aucとtmp_maeの値をファイルに出力
        std::ofstream ofs_auc(directory_name_of_auc + "/" + output_file_name_of_auc, std::ios::app);
        std::ofstream ofs_mae(directory_name_of_mae + "/" + output_file_name_of_mae, std::ios::app);
        if (ofs_auc && ofs_mae) {
            for (int k = 0; k < rs::missing_pattern; ++k) {
                ofs_auc << tmp_auc[k] << "\t";
                ofs_mae << tmp_mae[k] << "\t";
            }
            for (int i = 0; i < (int)parameters_.size(); i++) {
                ofs_auc << std::setprecision(10) << parameters_[i] << "\t";
                ofs_mae << std::setprecision(10) << parameters_[i] << "\t";
            }
            ofs_auc << std::endl;
            ofs_mae << std::endl;
            std::cout << "tmp_aucとtmp_maeの値をファイルに追記しました。" << std::endl;
        } else {
            std::cerr << "ファイルに書き込めませんでした。" << std::endl;
        }
    }
}

void Recom::output_high_score_in_tally_result() {
    std::string result = std::filesystem::current_path();
    result = result + "/../../RESULT";
    // 入力元のディレクトリ名
    std::string directory_name_of_auc = result + "/AUC";
    std::string directory_name_of_mae = result + "/MAE";
    directory_name_of_auc += "/summay";
    directory_name_of_mae += "/summay";

    // 入力元のファイル名
    std::string input_file_name_of_auc = method_name_ + "_" + rs::data_name + std::to_string(num_missing_value_) + "_summary" + ".txt";
    std::string input_file_name_of_mae = method_name_ + "_" + rs::data_name + std::to_string(num_missing_value_) + "_summary" + ".txt";

    // ここからAUC
    std::ifstream ifs_auc(directory_name_of_auc + "/" + input_file_name_of_auc);
    if (!ifs_auc) {
        std::cerr << "ファイルopen失敗: " << directory_name_of_auc + input_file_name_of_auc << std::endl;
    } else {
        std::cerr << "ファイルopen成功: " << directory_name_of_auc + input_file_name_of_auc << std::endl;
        // AUCのファイルの読み取り
        double best_sum_auc = 0.0;
        Vector best_auc_line(rs::missing_pattern + (int)parameters_.size());
        for (std::string str; std::getline(ifs_auc, str);) {
            std::istringstream stream(str);
            int j = 0;
            double tmp_sum_auc = 0;
            Vector tmp_auc_line(rs::missing_pattern + (int)parameters_.size());
            for (std::string tmp; getline(stream, tmp, '\t'); j++) {
                tmp_auc_line[j] = std::stod(tmp);
                if (j < rs::missing_pattern) tmp_sum_auc += std::stod(tmp);
            }
            if (tmp_sum_auc > best_sum_auc) {
                best_sum_auc = tmp_sum_auc;
                best_auc_line = tmp_auc_line;
            }
        }
        // 最高AUCの値とパラメータをファイルに出力
        std::ofstream ofs_auc(directory_name_of_auc + "/../" + rs::data_name + "_" + method_name_ + "_result.txt", std::ios::app);
        if (ofs_auc) {
            ofs_auc << num_missing_value_ << "\t";
            for (int k = 0; k < rs::missing_pattern + (int)parameters_.size(); ++k) {
                ofs_auc << best_auc_line[k] << "\t";
            }
            ofs_auc << std::endl;
            std::cout << "aucのresultファイルに追記しました。" << std::endl;
        } else {
            std::cerr << "ファイルに書き込めませんでした。" << std::endl;
        }
    }

    // ここからMAE
    std::ifstream ifs_mae(directory_name_of_mae + "/" + input_file_name_of_mae);
    if (!ifs_mae) {
        std::cerr << "ファイルopen失敗: " << directory_name_of_mae + input_file_name_of_mae << std::endl;
    } else {
        std::cerr << "ファイルopen成功: " << directory_name_of_mae + input_file_name_of_mae << std::endl;
        // MAEのファイルの読み取り
        double best_sum_mae = DBL_MAX;
        Vector best_mae_line(rs::missing_pattern + (int)parameters_.size());
        for (std::string str; std::getline(ifs_mae, str);) {
            std::istringstream stream(str);
            int j = 0;
            double tmp_sum_mae = 0;
            Vector tmp_mae_line(rs::missing_pattern + (int)parameters_.size());
            for (std::string tmp; getline(stream, tmp, '\t'); j++) {
                tmp_mae_line[j] = std::stod(tmp);
                if (j < rs::missing_pattern) tmp_sum_mae += std::stod(tmp);
            }
            if (tmp_sum_mae < best_sum_mae) {
                best_sum_mae = tmp_sum_mae;
                best_mae_line = tmp_mae_line;
            }
        }
        // 最高AUCの値とパラメータをファイルに出力
        std::ofstream ofs_mae(directory_name_of_mae + "/../" + rs::data_name + "_" + method_name_ + "_result.txt", std::ios::app);
        if (ofs_mae) {
            ofs_mae << num_missing_value_ << "\t";
            for (int k = 0; k < rs::missing_pattern + (int)parameters_.size(); ++k) {
                ofs_mae << best_mae_line[k] << "\t";
            }
            ofs_mae << std::endl;
            std::cout << "maeのresultファイルに追記しました。" << std::endl;
        } else {
            std::cerr << "ファイルに書き込めませんでした。" << std::endl;
        }
    }
}

std::vector<std::string> mkdir(std::vector<std::string> methods, int num_missing_value) {
    std::vector<std::string> v;
    std::string c_p = std::filesystem::current_path();
    c_p = c_p + "/../../RESULT/" + methods[0];
    mkdir(c_p.c_str(), 0755);
    for (int i = 0; i < (int)methods.size(); i++) {
        std::string d = c_p + "/" + methods[i] + "_" + rs::data_name + std::to_string(num_missing_value);
        mkdir(d.c_str(), 0755);
        // ROCフォルダ作成
        const std::string roc = d + "/ROC";
        mkdir(roc.c_str(), 0755);
        // 選ばれるROCファイルをまとめるフォルダ作成
        const std::string choice = roc + "/choice";
        mkdir(choice.c_str(), 0755);
        v.push_back(d);
    }
    return v;
}

std::vector<std::string> mkdir_result(std::vector<std::string> dirs, std::vector<double> parameters, int num_missing_value) {
    std::vector<std::string> v;
    std::string c_p = std::filesystem::current_path();
    c_p = c_p + "/../../RESULT/" + dirs[0];
    mkdir(c_p.c_str(), 0755);
    for (int i = 0; i < (int)dirs.size(); i++) {
        std::string d = c_p + "/" + dirs[i] + "_" + rs::data_name + std::to_string(num_missing_value);
        mkdir(d.c_str(), 0755);
        std::string mf_parameters = "";
        for (int i = 0; i < (int)parameters.size(); i++) {
            std::ostringstream oss;
            oss << std::setprecision(10) << parameters[i];
            std::string p(oss.str());
            mf_parameters += p;
            if (i < (int)parameters.size() - 1) mf_parameters += "_";
        }
        d += "/" + mf_parameters;
        mkdir(d.c_str(), 0755);
        // ROCフォルダ作成
        const std::string roc = d + "/ROC";
        mkdir(roc.c_str(), 0755);
        // 選ばれるROCファイルをまとめるフォルダ作成
        const std::string choice = roc + "/choice";
        mkdir(choice.c_str(), 0755);
        // diffフォルダ作成
        const std::string diff = d + "/diff";
        mkdir(diff.c_str(), 0755);
        v.push_back(d);
    }
    return v;
}

std::string append_current_time_if_test(std::string method) {
#if defined TEST
    // 現在の日付と時間を取得
    time_t now = time(0);
    tm *timeinfo = localtime(&now);
    // 月、日、時、分を文字列に変換
    char date_time_str[20];
    strftime(date_time_str, sizeof(date_time_str), "%m%d-%H%M", timeinfo);
    return method + "-" + std::string(date_time_str);
#else
    return method;
#endif
}

bool check_command_args(int argc, char *argv[]) {
    bool result = false;
    if (argc != 3) {
        std::cerr << "コマンドライン引数の数が正しくありません\n開始潜在次元%, 終了潜在次元%\n"
                  << "例: xxx.out 0 5" << std::endl;
        result = true;
    }

    int latent_dimensions_size = sizeof(rs::latent_dimensions);

    int latent_dimension_numbers[2];
    latent_dimension_numbers[0] = std::stoi(argv[1]);
    latent_dimension_numbers[1] = std::stoi(argv[2]);

    if (latent_dimension_numbers[0] > latent_dimensions_size || latent_dimension_numbers[1] > latent_dimensions_size ||
        latent_dimension_numbers[0] > latent_dimension_numbers[1] || latent_dimension_numbers[0] < 0 || latent_dimension_numbers[1] < 0) {
        std::cerr << "コマンドライン引数を正しく指定してください\n開始潜在次元%, 終了潜在次元%\n"
                  << "例: xxx.out 0 5" << std::endl;
        result = true;
    }
    return result;
}