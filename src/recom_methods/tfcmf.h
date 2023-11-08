#include "../recom_system_base/tfc_recom.h"

class TFCMF : virtual public TFCRecom {
    // private:

   protected:
    // 正則化パラメータ
    double reg_parameter_;
    // 学習率
    double learning_rate_;
    // 潜在次元
    int latent_dimension_;
    // ユーザー行列とアイテム行列
    Tensor user_factors_, item_factors_;
    Tensor prev_user_factors_, prev_item_factors_;
    double *user_factor_values_,*item_factor_values_;

   public:
    TFCMF(int missing_pattern);
    void set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda, double reg_parameter,
                        double learning_rate);
    void calculate_factors() override;
    void set_initial_values(int &seed) override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};