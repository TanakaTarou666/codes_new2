#include "../math_utils/dss_tensor.h"
#include "../recom_system_base/fm_base.h"

class FMWithALS : virtual public FMBase {
   protected:
    // 潜在次元
    int latent_dimension_;
    // 正則化パラメータ
    double reg_parameter_;
    double w0_, prev_w0_;
    Vector w_, prev_w_, e_;
    Matrix v_, prev_v_, q_;
    DSSTensor x_;

   public:
    FMWithALS(int missing_count);
    void set_parameters(double latent_dimension_percentage,double reg_parameter);
    void train() override;
    void set_initial_values(int &seed);
    void precompute();
    void calculate_Wo_w_v();
    double calculate_objective_value();
    bool calculate_convergence_criterion();
    void calculate_prediction();
};