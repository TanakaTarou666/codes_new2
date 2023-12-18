#include <iomanip>

#define ARTIFICIALITY
//#define TEST
//#define MOVIE
//#define BOOK

namespace rs {

#if defined TEST
#define ARTIFICIALITY
#endif

#if !defined ARTIFICIALITY
const double latent_dimensions[] = {1.0, 2.0, 3.0};
const double reg_parameters[] = {0.01, 0.05, 0.09, 0.13};
const double learning_rates[] = {0.001};
const int cluster_size[] = {2, 3, 4, 5};
const double fuzzifier_em[] = {1.001, 1.1, 1.5, 2.3};
const double fuzzifier_lambda[] = {1000};
const double convergence_criteria = 1.0e-4;
const int missing_pattern = 5;
const int num_initial_values = 4;
const int steps = 2000;
#endif

#if defined TEST
const std::string data_name = "artificiality";
const double latent_dimensions[] = {5.0};
const double reg_parameters[] = {0.01};
const double learning_rates[] = {0.001};
const int cluster_size[] = {2};
const double fuzzifier_em[] = {1.001};
const double fuzzifier_lambda[] = {1000};
const double convergence_criteria = 0.011;
const int missing_pattern = 4;
const int num_initial_values = 4;
const int steps = 2000;
const int num_users = 80;
const int num_items = 100;
const int num_samples=8000;
const double max_value = 4.0;
const int start_missing_valu = 5000;
const int end_missing_valu = 5000;
const int step_missing_valu = 1000;
#elif defined ARTIFICIALITY
const std::string data_name = "artificiality";
const double latent_dimensions[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0};
const double reg_parameters[] = {0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19};
const double learning_rates[] = {0.001};
const int cluster_size[] = {2, 3, 4, 5};
const double fuzzifier_em[] = {1.001, 1.01, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5};
const double fuzzifier_lambda[] = {10, 100, 1000};
const double convergence_criteria = 0.011;
const int missing_pattern = 4;
const int num_initial_values = 4;
const int steps = 2000;
const int num_users = 80;
const int num_items = 100;
const int num_samples=8000;
const double max_value = 4.0;
const int start_missing_valu = 5000;
const int end_missing_valu = 5000;
const int step_missing_valu = 500;
#elif defined BOOK
const std::string data_name = "book";
const int num_users = 1091;
const int num_items = 2248;
const int num_samples=35179;
const double max_value = 10.0;
const int start_missing_valu = 20000;
const int end_missing_valu = 20000;
const int step_missing_valu = 500;
#elif defined MOVIE
const std::string data_name = "movie";
const int num_users = 905;
const int num_items = 684;
const int num_samples=277546;
const double max_value = 5.0;
const int start_missing_valu = 20000;
const int end_missing_valu = 20000;
const int step_missing_valu = 2000;
// #elif defined MOVIE100K
// const std::string data_name = "movie100k";
// const int num_users = 874;
// const int num_items = 598;
// const double max_value = 5.0;
// const int start_missing_valu = 40000;
// const int end_missing_valu = 40000;
// const int step_missing_valu = 2000;
#elif defined MOVIE10M
const std::string data_name = "movie10m";
const int num_users = 1299;
const int num_items = 1695;
const int num_samples=1022610;
const double max_value = 5.0;
const int start_missing_valu = 20000;
const int end_missing_valu = 20000;
const int step_missing_valu = 2000;
#elif defined LIBIMSETI
const std::string data_name = "libimseti";
const int num_users = 866;
const int num_items = 1156;
const int num_samples=400955;
const double max_value = 10.0;
const int start_missing_valu = 20000;
const int end_missing_valu = 20000;
const int step_missing_valu = 2000;
#elif defined SUSHI
const std::string data_name = "sushi";
const int num_users = 5000;
const int num_items = 100;
const int num_samples=50000;
const double max_value = 5.0;
const int start_missing_valu = 20000;
const int end_missing_valu = 20000;
const int step_missing_valu = 2000;
#elif defined JESTER
const std::string data_name = "jester";
const int num_users = 2916;
const int num_items = 140;
const int num_samples=373338;
const double max_value = 21.0;
const int start_missing_valu = 20000;
const int end_missing_valu = 20000;
const int step_missing_valu = 2000;
#elif defined NETFLIX
const std::string data_name = "netflix";
const int num_users = 542;
const int num_items = 4495;
const int num_samples=1291999;
const double max_value = 5.0;
const int start_missing_valu = 100000;
const int end_missing_valu = 100000;
const int step_missing_valu = 2000;
#endif

// 入力するデータの場所
const std::string input_data_name = "data/sparse_" + data_name + "_" + std::to_string(num_users) + "_" + std::to_string(num_items) + ".txt";
}  // namespace rs