#include <float.h>
#include <sys/stat.h>

#include <cfloat>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "../math_utils/tensor.h"
#include "../math_utils/sparse_matrix.h"
#include "../math_utils/sparse_vector.h"

#ifndef __RECOM__
#define __RECOM__

#include "data_manager.h"

class Recom {
   private:
    // 欠損のさせ方を決めるシード値
    int seed_ = 0;
    
   protected:
    // 欠損数
    int num_missing_value_;
    std::vector<std::string> dirs_;
    std::vector<double> parameters_;
    std::string method_name_;
    int sum_users_items_= rs::num_users+rs::num_items;
    // 欠損前データ
    SparseMatrix sparse_correct_data_;
    
    // 欠損後データ
    SparseMatrix sparse_missing_data_;
    double *sparse_missing_data_values_;
    int *sparse_missing_data_row_pointers_;
    int sparse_missing_data_row_nnzs_[rs::num_users];
    int *sparse_missing_data_col_indices_;
    int missing_num_samples_;
    
    // エラーの検知
    bool error_detected_;
    // 欠損させた箇所，類似度
    Matrix missing_data_indices_;
    // 欠損させた箇所のスパースデータの列番号
    Vector sparse_missing_data_cols_;
    // 予測評
    Vector prediction_;
    double prev_objective_value_, objective_value_;
    // 予測値評価
    Vector mae_, auc_;
    Vector tp_fn_, fp_tn_;


   public:
    // ユーザ数，アイテム数，欠損数，欠損パターン
    Recom(int num_missing_value);
    void input(std::string);
    void revise_missing_values(void);
    // 実際の計算
    virtual void train();
    virtual void set_initial_values(int seed);
    virtual void calculate_factors();
    virtual double calculate_objective_value();
    virtual bool calculate_convergence_criterion();
    virtual void calculate_prediction();
    // MAEの計算，textに保存
    void calculate_mae(int current_missing_pattern);
    // ROCで必要な値をtextに保存
    void calculate_roc(int current_missing_pattern);
    // ROCの横軸の値で小さい順にソート
    void sort(Vector &fal, Vector &tru, int index);
    // AUCの計算，text1に読み込むROCファイル，text2に平均AUCを保存
    void precision_summury();
    // 結果の集計
    void tally_result();
    void output_high_score_in_tally_result();
};

// 結果を出力するフォルダを作成
std::vector<std::string> mkdir(std::vector<std::string> methods, int num_missing_value);
std::vector<std::string> mkdir_result(std::vector<std::string> dirs, std::vector<double> parameters, int num_missing_value);

// main.cxxで使う関数
std::string append_current_time_if_test(std::string method);
bool check_command_args(int argc, char *argv[]);

#endif
