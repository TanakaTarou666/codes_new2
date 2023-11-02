#include <iomanip>

#define BOOK

const double convergence_criteria = 0.001;
const int missing_pattern = 5;
const int num_initial_values = 5;
const int steps=2000;

#ifdef ARTIFICIALITY
const std::string data_name = "artificiality";
const int num_users = 80;
const int num_items = 100;
const double max_value = 4.0;
const int start_missing_valu = 5000;
const int end_missing_valu = 5000;
const int step_missing_valu = 500;
#elif defined BOOK
const std::string data_name = "book";
const int num_users = 1091;
const int num_items = 2248;
const double max_value = 10.0;
const int start_missing_valu = 15000;
const int end_missing_valu = 15000;
const int step_missing_valu = 500;
#endif

// 入力するデータの場所
const std::string input_data_name = "data/sparse_" + data_name + "_" + std::to_string(num_users) + "_" + std::to_string(num_items) + ".txt";