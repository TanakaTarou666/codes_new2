#files=("mf" "tfcmf" "qfcmf")
#files=("fm_sgd" "tfcfm_sgd" "qfcfm_sgd") #
files=("tfcmf") #
# 制御用のFIFOファイルを作成
# 2つのコアで並列処理を実行
for file in "${files[@]}"; do #
  { #
    for i in {0..5}; do #
      { #
        rm -f ".out/${file}_${i}" #
        make ".out/${file}_${i}" ARG=${i} #
        # コア指定
        taskset -c $((0+i))-$((0+i)) ".out/${file}_$((0+i))" $i $i #
      } & #
    done #
    wait #
  } #
done #

# 全てのジョブが完了するまで待機
wait #
