#include "recom.h"

Recom::Recom(int num_missing_value)
    : tp_fn_((int)max_value * 100, 0.0, "all"),
      fp_tn_((int)max_value * 100, 0.0, "all"),
      num_missing_value_(num_missing_value),
      sparse_missing_data_(num_users, num_items),
      sparse_correct_data_(num_users, num_items),
      missing_data_indices_(num_missing_value, 2),
      prediction_(num_missing_value, 0, "all"),
      sparse_missing_data_cols_(num_missing_value),
      mae_(missing_pattern),
      auc_(missing_pattern) {}

void Recom::train() { return; }

void Recom::input(std::string file_name) {
    std::ifstream ifs(file_name);
    if (!ifs) {
        std::cerr << "directory_name:" << file_name << ": could not open." << std::endl;
        exit(1);
    }

    int *row_pointers = new int[num_users + 1]();
    int row_sizes[num_users];
    std::vector<std::vector<double>> correct_data;
    std::vector<std::vector<int>> correct_data_index;
    row_pointers[0] = 0;
    for (int cnt = 0; cnt < num_users; cnt++) {
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

    int *col_indices = new int[row_pointers[num_users]]();
    double *values = new double[row_pointers[num_users]]();
    int i = 0;
    for (int cnt = 0; cnt < num_users; cnt++) {
        for (int ell = 0; ell < row_sizes[cnt]; ell++) {
            col_indices[i] = correct_data_index[cnt][ell];
            values[i] = correct_data[cnt][ell];
            i++;
        }
    }

    sparse_correct_data_.set_col_indices(col_indices);
    sparse_correct_data_.set_values(values);
    sparse_correct_data_.set_nnz(row_pointers[num_users]);
    sparse_missing_data_ = sparse_correct_data_;
}

void Recom::revise_missing_values(void) {
    sparse_missing_data_ = sparse_correct_data_;
    int tmprow, tmpcol;
    for (int m = 0; m < num_missing_value_;) {
        /****乱数生成****/
        std::mt19937_64 mt;
        mt.seed(seed_);
        std::uniform_int_distribution<> randRow(0, num_users - 1);
        // ランダムに行番号生成
        tmprow = randRow(mt);
        int col_size = sparse_correct_data_.get_row_pointers()[tmprow + 1] - sparse_correct_data_.get_row_pointers()[tmprow];
        std::uniform_int_distribution<> randCol(0, col_size - 1);
        // ランダムに列番号生成
        tmpcol = randCol(mt);
        // データ行すべて欠損させないように,一行5要素は残す
        int c = 0;
        for (int i = 0; i < col_size; i++)
            if (sparse_missing_data_(tmprow, i) == 0) c++;
        // 既に欠損していない場合
        if (sparse_missing_data_(tmprow, tmpcol) > 0 && col_size - c > 5) {
            // 要素を0にする
            sparse_missing_data_(tmprow, tmpcol) = 0;
            // 欠損した行番号を保存
            missing_data_indices_[m][0] = tmprow;
            // 欠損した列番号を保存
            missing_data_indices_[m][1] = sparse_missing_data_(tmprow, tmpcol, "index");
            // スパースデータの列番号を保存
            sparse_missing_data_cols_[m] = tmpcol;
            m++;
        }
        seed_++;
    }
    sparse_missing_data_values_=sparse_missing_data_.get_values();
    sparse_missing_data_row_pointers_=sparse_missing_data_.get_row_pointers();
    sparse_missing_data_col_indices_=sparse_missing_data_.get_col_indices();
    return;
}

void Recom::calculate_mae(int current_missing_pattern) {
    double result = 0.0;
    for (int m = 0; m < num_missing_value_; m++) {
        result += fabs(sparse_correct_data_(missing_data_indices_[m][0], sparse_missing_data_cols_[m]) - prediction_[m]);
    }
    mae_[current_missing_pattern] = result / (double)num_missing_value_;

    std::ofstream ofs(dirs_[0] + "/" + method_name_ + "MAE.txt", std::ios::app);
    ofs << num_missing_value_ << "\t" << seed_ << "\t" << current_missing_pattern << "\t" << std::fixed << std::setprecision(10)
        << result / (double)num_missing_value_ << std::endl;
    ofs.close();
    return;
}

void Recom::calculate_roc(int current_missing_pattern) {
    for (int index = 1; index < (int)max_value * 100; index++) {
        double TP = 0.0, FP = 0.0, FN = 0.0, TN = 0.0;
        // 閾値の設定
        double siki = (double)index / 100.0;
        for (int m = 0; m < num_missing_value_; m++) {
            // 正解値が閾値以上かつ，予測値が閾値以上場合
            if ((siki <= sparse_correct_data_(missing_data_indices_[m][0], sparse_missing_data_cols_[m])) && (siki <= prediction_[m])) TP += 1.0;
            // 正解値が閾値を下回ったかつ，予測値が閾値上回った場合
            else if ((siki > sparse_correct_data_(missing_data_indices_[m][0], sparse_missing_data_cols_[m])) && (siki <= prediction_[m]))
                FP += 1.0;
            // 正解値が閾値上回ったかつ，予測値が閾値を下回った場合
            else if ((siki <= sparse_correct_data_(missing_data_indices_[m][0], sparse_missing_data_cols_[m])) && (siki > prediction_[m]))
                FN += 1.0;
            // それ以外
            else if ((siki > sparse_correct_data_(missing_data_indices_[m][0], sparse_missing_data_cols_[m])) && (siki > prediction_[m]))
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
    std::string ROC_STR = dirs_[0] + "/ROC/choice/" + method_name_ + "ROC" + std::to_string(num_missing_value_) + "_" + std::to_string(current_missing_pattern) + "sort.txt";
    // ROCでプロットする点の数
    int max_index = (int)max_value * 100;
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
    int max = (int)max_value * 100;
    for (int method = 0; method < (int)dirs_.size(); method++) {
        double rocarea = 0.0;
        for (int x = 0; x < missing_pattern; x++) {
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
        for (int i = 0; i < missing_pattern; i++) {
            ofs << std::fixed << std::setprecision(10) << mae_[i] << "\t" << auc_[i] << std::endl;
        }
    }
    seed_=0;
    return;
}

std::vector<std::string> mkdir(std::vector<std::string> methods, int num_missing_value) {
    std::vector<std::string> v;
    std::string c_p = std::filesystem::current_path();
    c_p = c_p + "/../../RESULT/" + methods[0];
    mkdir(c_p.c_str(), 0755);
    for (int i = 0; i < (int)methods.size(); i++) {
        std::string d = c_p + "/" + methods[i] + "_" + data_name + std::to_string(num_missing_value);
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
#if defined ARTIFICIALITY
    if (num_users > num_items) {
        parameters[0] = std::round(num_items * parameters[0] / 100);
    } else {
        parameters[0] = std::round(num_users * parameters[0] / 100);
    }
#else
    if (num_users > num_items) {
        parameters[0] = std::round(num_items * parameters[0] / 100);
    } else {
        parameters[0] = std::round(num_users * parameters[0] / 100);
    }
#endif
    std::vector<std::string> v;
    std::string c_p = std::filesystem::current_path();
    c_p = c_p + "/../../RESULT/" + dirs[0];
    mkdir(c_p.c_str(), 0755);
    for (int i = 0; i < (int)dirs.size(); i++) {
        std::string d = c_p + "/" + dirs[i] + "_" + data_name + std::to_string(num_missing_value);
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
