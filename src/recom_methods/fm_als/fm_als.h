#include "../../math_utils/dss_tensor.h"
#include "../../recom_system_base/fm_base.h"

class FMWithALS : virtual public FMBase {
   protected:
    // 潜在次元
    int latent_dimension_;
    // 正則化パラメータ
    double reg_parameter_;
    // 要素
    double w0_, prev_w0_;
    Vector w_, prev_w_, e_;
    Matrix v_, prev_v_, q_;
    // データ
    DSSTensor x_;
    SparseMatrix  transpose_x_;

   public:
    FMWithALS(int missing_count);
    void set_parameters(double latent_dimension_percentage,double reg_parameter);
    void set_initial_values(int seed) override;
    void precompute();
    void calculate_factors() override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};