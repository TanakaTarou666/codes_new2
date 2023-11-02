#include "../math_utils/dsd_tensor.h"
#include "../recom_system_base/tfc_recom.h"

class TFCFMWithALS : virtual public TFCRecom , virtual public Recom{
   protected:
    // 潜在次元
    int latent_dimension_;
    Vector w0_, prev_w0_;
    Matrix w_, prev_w_, e_;
    Tensor v_, prev_v_, q_;
    DSDTensor X_;

   public:
    TFCFMWithALS(int missing_count);
    void set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda);
    void train() override;
    void precompute();
    void calculate_Wo_w_v();
    void calculate_dissimilarities() override;
    void set_initial_values(int &seed);
    double calculate_objective_value();
    bool calculate_convergence_criterion();
    void calculate_prediction();
};