CXX = g++ -std=c++17 
CXXFLAGS = -O3 -mtune=native -march=native -mfpmath=both
#CXXFLAGS =  -g -Wall -Wextra  #debug用

ARG = test

math_utils = src/math_utils/vector.cxx \
            src/math_utils/matrix.cxx \
			src/math_utils/sparse_vector.cxx \
            src/math_utils/sparse_matrix.cxx \
            src/math_utils/tensor.cxx \
            src/math_utils/dss_tensor.cxx

# recom_system_base
normal_recom = src/recom_system_base/recom.cxx $(math_utils)
tfc_recom = src/recom_system_base/tfc_recom.cxx $(normal_recom)
qfc_recom = src/recom_system_base/qfc_recom.cxx src/recom_system_base/tfc_recom.cxx $(normal_recom)
fm_recom = src/recom_system_base/fm_base.cxx $(normal_recom)
tfcfm_recom = src/recom_system_base/tfc_recom.cxx src/recom_system_base/fm_base.cxx $(normal_recom)
qfcfm_recom = src/recom_system_base/qfc_recom.cxx src/recom_system_base/tfc_recom.cxx src/recom_system_base/fm_base.cxx $(normal_recom)

# mf
.out/mf_$(ARG): src/recom_methods/mf/mf.cxx main_recom/mf/mf_main.cxx $(normal_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/tfcmf_$(ARG): src/recom_methods/mf/tfcmf.cxx main_recom/mf/tfcmf_main.cxx $(tfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/qfcmf_$(ARG): src/recom_methods/mf/qfcmf.cxx main_recom/mf/qfcmf_main.cxx $(qfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@	

# wnmf	
.out/wnmf_$(ARG): src/recom_methods/wnmf/wnmf.cxx main_recom/wnmf/wnmf_main.cxx $(normal_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/tfcwnmf: src/recom_methods/wnmf/tfcwnmf.cxx main_recom/wnmf/tfcwnmf_main.cxx $(tfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@

# fm_als
.out/fm_als_$(ARG): src/recom_methods/fm_als/fm_als.cxx main_recom/fm_als/fm_als_main.cxx $(fm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/tfcfm_als_$(ARG): src/recom_methods/fm_als/tfcfm_als.cxx main_recom/fm_als/tfcfm_als_main.cxx $(tfcfm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@	
.out/qfcfm_als_$(ARG): src/recom_methods/fm_als/qfcfm_als.cxx main_recom/fm_als/qfcfm_als_main.cxx $(qfcfm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@	

# fm_sgd
.out/fm_sgd_$(ARG): src/recom_methods/fm_sgd/fm_sgd.cxx main_recom/fm_sgd/fm_sgd_main.cxx $(fm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@	
.out/tfcfm_sgd_$(ARG): src/recom_methods/fm_sgd/tfcfm_sgd.cxx main_recom/fm_sgd/tfcfm_sgd_main.cxx $(tfcfm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/qfcfm_sgd_$(ARG): src/recom_methods/fm_sgd/qfcfm_sgd.cxx main_recom/fm_sgd/qfcfm_sgd_main.cxx $(qfcfm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@	

# result 
.out/result : main_recom/result/result.cxx src/recom_methods/mf/mf.cxx src/recom_methods/mf/tfcmf.cxx src/recom_methods/mf/qfcmf.cxx src/recom_methods/fm_als/fm_als.cxx src/recom_methods/fm_sgd/fm_sgd.cxx src/recom_methods/fm_sgd/tfcfm_sgd.cxx src/recom_methods/fm_sgd/qfcfm_sgd.cxx $(qfcfm_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@


clean:
	rm -f .out/mf
	rm -f .out/tfcmf
	rm -f .out/qfcmf
	rm -f .out/tfcwnmf
	rm -f .out/fm_als
	rm -f .out/fm_sgd
	rm -f .out/tfcfm_sgd
	rm -f .out/tfcfm_als

.PHONY: clean
