#include "../../math_utils/dss_tensor.h"
#include "../../recom_system_base/fm_base.h"
#include "../../recom_system_base/tfc_recom.h"

class TFCFMWithSGD : virtual public FMBase, virtual public TFCRecom {
   protected:
    // 正則化パラメータ
    double reg_parameter_;
    // 学習率
    double learning_rate_;
    // 潜在次元
    int latent_dimension_;
    Vector w0_, prev_w0_;
    Matrix w_, prev_w_, e_;
    Tensor v_, prev_v_, q_;
    DSSTensor x_;

   public:
    TFCFMWithSGD(int missing_count);
    void set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda, double reg_parameter,
                        double learning_rate);
    void set_initial_values(int &seed) override;
    void calculate_factors() override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};