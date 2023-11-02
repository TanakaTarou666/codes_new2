CXX = g++
CXXFLAGS = -O3 -mtune=native -march=native -mfpmath=both
#CXXFLAGS =  -g -Wall -Wextra  #debug用

math_utils = src/math_utils/vector.cxx \
            src/math_utils/matrix.cxx \
            src/math_utils/sparse_matrix.cxx \
            src/math_utils/tensor.cxx \
            src/math_utils/dsd_tensor.cxx

# recom_system_base
normal_recom = src/recom_system_base/recom.cxx $(math_utils)
tfc_recom = src/recom_system_base/tfc_recom.cxx \
            src/recom_system_base/recom.cxx $(math_utils)

# mf
.out/mf: src/recom_methods/mf.cxx main_recom/mf_main.cxx $(normal_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/tfcmf: src/recom_methods/tfcmf.cxx main_recom/tfcmf_main.cxx $(tfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@

# nmf	
.out/tfcwnmf: src/recom_methods/tfcwnmf.cxx main_recom/tfcwnmf_main.cxx $(tfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@

# fm
.out/tfcfm_sgd: src/recom_methods/tfcfm_sgd.cxx main_recom/tfcfm_sgd_main.cxx $(tfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@
.out/tfcfm_als: src/recom_methods/tfcfm_als.cxx main_recom/tfcfm_als_main.cxx $(tfc_recom)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f .out/mf
	rm -f .out/tfcmf
	rm -f .out/tfcwnmf
	rm -f .out/tfcfm_sgd
	rm -f .out/tfcfm_als

.PHONY: clean
