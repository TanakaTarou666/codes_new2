#include <float.h>
#include <sys/stat.h>
#include <cfloat>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "../math_utils/sparse_matrix.h"
#include "data_manager.h"

#ifndef __RECOM__
#define __RECOM__

class Recom {
   private:
    // 欠損のさせ方を決めるシード値
    int seed_;
    // 予測値評価
    Vector mae_, auc_;
    Vector tp_fn_, fp_tn_;

   protected:
    //欠損数
    int num_missing_value_;
    std::vector<std::string> dirs_;
    std::vector<double> parameters_;
    std::string method_name_;
    // 欠損後データ
    SparseMatrix sparse_missing_data_;
    // 欠損前データ
    SparseMatrix sparse_correct_data_;
    double *sparse_missing_data_values_;
    int *sparse_missing_data_row_pointers_,*sparse_missing_data_col_indices_;
    // エラーの検知
    bool error_detected_;
    // 欠損させた箇所，類似度
    Matrix missing_data_indices_;
    // 欠損させた箇所のスパースデータの列番号
    Vector sparse_missing_data_cols_;
    // 予測評
    Vector prediction_;
    double prev_objective_value_ ,objective_value_;


   public:
    // ユーザ数，アイテム数，欠損数，欠損パターン
    Recom(int num_missing_value);
    virtual void train();
    void input(std::string);
    void revise_missing_values(void);
    // MAEの計算，textに保存
    void calculate_mae(int current_missing_pattern);
    // ROCで必要な値をtextに保存
    void calculate_roc(int current_missing_pattern);
    // ROCの横軸の値で小さい順にソート
    void sort(Vector &fal, Vector &tru, int index);
    // AUCの計算，text1に読み込むROCファイル，text2に平均AUCを保存
    void precision_summury();
};

std::vector<std::string> mkdir(std::vector<std::string> methods, int num_missing_value);
// 結果を出力するフォルダを作成
std::vector<std::string> mkdir_result(std::vector<std::string> dirs,std::vector<double> parameters, int num_missing_value);

#endif
