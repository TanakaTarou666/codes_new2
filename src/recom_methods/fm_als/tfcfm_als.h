#include "../../math_utils/dss_tensor.h"
#include "../../recom_system_base/tfc_recom.h"
#include "../../recom_system_base/fm_base.h"

class TFCFMWithALS : virtual public FMBase, virtual public TFCRecom{
   protected:
    // 潜在次元
    int latent_dimension_;
    Vector w0_, prev_w0_;
    Matrix w_, prev_w_, e_;
    Tensor v_, prev_v_, q_;
    DSSTensor x_;

   public:
    TFCFMWithALS(int missing_count);
    void set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda);
    void set_initial_values(int seed) override;
    void precompute() override;
    void calculate_factors() override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};