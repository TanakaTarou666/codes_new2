#include "recom.h"

#ifndef __MFBASE__
#define __MFBASE__

class MFBase : virtual public Recom {
   protected:
    // 正則化パラメータ
    double reg_parameter_;
    // 学習率
    double learning_rate_;
    // 潜在次元
    int latent_dimension_;
    // ユーザー行列とアイテム行列
    Matrix user_factors_, item_factors_;
    Matrix prev_user_factors_, prev_item_factors_;
    double *user_factor_values_,*item_factor_values_;
    
    
   public:
    MFBase(int missing_count);
    void set_parameters(double latent_dimension_percentage, double learning_rate, double reg_parameter);
    void set_initial_values(int seed) override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};

#endif