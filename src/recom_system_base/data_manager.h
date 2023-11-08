#include <iomanip>

// #define ARTIFICIALITY
// #define TEST

#define BOOK

#ifdef TEST
const double convergence_criteria = 0.01;
const int missing_pattern = 1;
const int num_initial_values = 1;
const int steps = 20000;
#else
#ifdef ARTIFICIALITY
const double convergence_criteria = 0.011;
#else
const double convergence_criteria = 1.0e-5;
#endif
const int missing_pattern = 5;
const int num_initial_values = 5;
const int steps = 2000;
#endif

#ifdef ARTIFICIALITY
const std::string data_name = "artificiality";
const int num_users = 80;
const int num_items = 100;
const double max_value = 4.0;
#ifdef TEST
const int start_missing_valu = 5000;
const int end_missing_valu = 7000;
const int step_missing_valu = 2000;
#else
const int start_missing_valu = 5000;
const int end_missing_valu = 7000;
const int step_missing_valu = 500;
#endif
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