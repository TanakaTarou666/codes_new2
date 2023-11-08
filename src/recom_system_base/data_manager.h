#include <iomanip>

// #define ARTIFICIALITY
// #define TEST

#define BOOK

namespace rs {

#ifdef TEST
const double latent_dimensions[] = {5.0};
const double reg_parameters[] = {0.01,0.09};
const double learning_rates[] = {0.001};
const int cluster_size[] = {2,5};
const double fuzzifier_em[] = {1.001};
const double fuzzifier_Lambda[] = {1000};
const double convergence_criteria = 0.01;
const int missing_pattern = 1;
const int num_initial_values = 1;
const int steps = 20000;
#elif ARTIFICIALITY
const double latent_dimensions[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
const double reg_parameters[] = {0.01, 0.05, 0.09, 0.13};
const double learning_rates[] = {0.001};
const int cluster_size[] = {2, 4, 5, 8, 9};
const double fuzzifier_em[] = {1.001};
const double fuzzifier_Lambda[] = {1000};
const double convergence_criteria = 0.011;
const int missing_pattern = 5;
const int num_initial_values = 5;
const int steps = 2000;
#else
const double latent_dimensions[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
const double reg_parameters[] = {0.01, 0.05, 0.09, 0.13};
const double learning_rates[] = {0.001};
const int cluster_size[] = {2, 4, 5, 8, 9};
const double fuzzifier_em[] = {1.001};
const double fuzzifier_Lambda[] = {1000};
const double convergence_criteria = 1.0e-5;
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
const std::string input_data_name = "data/sparse_" + data_name + "_" +
                                    std::to_string(num_users) + "_" +
                                    std::to_string(num_items) + ".txt";
}  // namespace rs