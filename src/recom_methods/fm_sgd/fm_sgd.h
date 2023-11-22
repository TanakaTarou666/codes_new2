#include "../../math_utils/dss_tensor.h"
#include "../../recom_system_base/fm_base.h"

class FMWithSGD : virtual public FMBase {
   protected:
    // 正則化パラメータ
    double reg_parameter_;
    // 学習率
    double learning_rate_;
    // 潜在次元
    int latent_dimension_;
    // 要素
    double w0_, prev_w0_;
    Vector w_, prev_w_;
    Matrix v_, prev_v_;
    // データ
    DSSTensor x_;

   public:
    FMWithSGD(int missing_count);
    void set_parameters(double latent_dimension_percentage, double reg_parameter, double learning_rate);
    void set_initial_values(int seed) override;
    void calculate_factors() override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};