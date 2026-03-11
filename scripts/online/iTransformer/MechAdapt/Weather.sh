#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./logs ./logs/online ./results


seq_len=512
data=Weather
datasets=("$data")

model_name=iTransformer
online_method=MechAdapt
tune_mode=down_up

feature=MS
target_variable=OT


ema=0
mid=100
batch_size=32

pred_lens=(96)


get_btl () {
  local pl="$1"
  if [[ "$pl" == "24" ]]; then
    echo 32
  elif [[ "$pl" == "48" ]]; then
    echo 48
  else
    echo 32   # 96
  fi
}

get_lr () {
  local pl="$1"
  if [[ "$pl" == "96" ]]; then
    echo 0.003
  elif [[ "$pl" == "48" ]]; then
    echo 0.001
  else
    echo 0.001 # 24
  fi
}

get_online_lr () {
  local pl="$1"
  if [[ "$pl" == "96" ]]; then
    echo 0.0000001
  elif [[ "$pl" == "48" ]]; then
    echo 0.0000001
  else
    echo 0.00000001 # 24
  fi
}

get_bs () {
  local pl="$1"
  if [[ "$pl" == "24" ]]; then
    echo 8
  elif [[ "$pl" == "48" ]]; then
    echo 16
  else
    echo 32   # 96
  fi
}


for pred_len in "${pred_lens[@]}"; do
  btl="$(get_btl "$pred_len")"
  learning_rate="$(get_lr "$pred_len")"
  batch_size="$(get_bs "$pred_len")"
  online_learning_rate="$(get_online_lr "$pred_len")"

  suffix="_lr${learning_rate}_onlinelr${online_learning_rate}"
  filename="logs/online/${model_name}_${online_method}_mid${mid}_share_fulltune_${tune_mode}_${data}_${pred_len}_btl${btl}${suffix}.log"

  python -u run.py \
    --dataset "$data" --border_type online --batch_size "$batch_size" \
    --model "$model_name" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --tune_mode "$tune_mode" \
    --features $feature \
    --target $target_variable \
    --target_variate 20 \
    --concept_dim "$mid" \
    --val_online_lr --diff_online_lr --only_test \
    --itr 3 --skip "$filename" --pretrain \
    --save_opt --online_method "$online_method" \
    --bottleneck_dim "$btl" \
    --online_learning_rate "$online_learning_rate" \
    --learning_rate "$learning_rate" --lradj type3 >> "$filename" 2>&1
done



mkdir -p "results/${data}"
out_csv="results/${data}/metrics_${data}_${model_name}_${online_method}.csv"
echo "data,model,online_method,pred_len,rmse_mean,rmse_std,mae_mean,mae_std,smape_mean,smape_std,log_file" > "$out_csv"

for pred_len in "${pred_lens[@]}"; do
  btl="$(get_btl "$pred_len")"
  learning_rate="$(get_lr "$pred_len")"
  online_learning_rate="$(get_online_lr "$pred_len")"

  suffix="_lr${learning_rate}_onlinelr${online_learning_rate}"
  logfile="logs/online/${model_name}_${online_method}_mid${mid}_share_fulltune_${tune_mode}_${data}_${pred_len}_btl${btl}${suffix}.log"

  if [[ ! -f "$logfile" ]]; then
    echo "warn: missing $logfile"
    echo "${data},${model_name},${online_method},${pred_len},,,,,,,,$logfile" >> "$out_csv"
    continue
  fi

  awk -v data="$data" -v model="$model_name" -v method="${online_method:-only_test}" \
      -v pl="$pred_len" -v logf="$logfile" '
    BEGIN{
      n_rmse=0; s_rmse=0; ss_rmse=0
      n_mae=0;  s_mae=0;  ss_mae=0
      n_sm=0;   s_sm=0;   ss_sm=0
      saw_rmse=0
    }

    $0 ~ /rmse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
      saw_rmse=1
      match($0,/rmse:[[:space:]]*([0-9.eE+\-]+)/,r); rm=r[1]+0
      match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a);  ma=a[1]+0
      s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++
      s_mae +=ma; ss_mae +=ma*ma; n_mae++
      if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
        sm = s[1]+0
        s_sm += sm; ss_sm += sm*sm; n_sm++
      }
      next
    }

    !saw_rmse && $0 ~ /mse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
      match($0,/mse:[[:space:]]*([0-9.eE+\-]+)/,m); ms=m[1]+0
      match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a);  ma=a[1]+0
      rm=sqrt(ms)
      s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++
      s_mae +=ma; ss_mae +=ma*ma; n_mae++
      if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
        sm = s[1]+0
        s_sm += sm; ss_sm += sm*sm; n_sm++
      }
      next
    }

    END{
      if (n_rmse>0 && n_mae>0) {
        m_rmse=s_rmse/n_rmse; v_rmse=ss_rmse/n_rmse - m_rmse*m_rmse; if (v_rmse<0) v_rmse=0; sd_rmse=sqrt(v_rmse)
        m_mae =s_mae /n_mae;  v_mae =ss_mae /n_mae  - m_mae*m_mae;  if (v_mae <0) v_mae=0; sd_mae=sqrt(v_mae)

        if (n_sm>0) {
          m_sm=s_sm/n_sm; v_sm=ss_sm/n_sm - m_sm*m_sm; if (v_sm<0) v_sm=0; sd_sm=sqrt(v_sm)
          printf("%s,%s,%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n",
                 data, model, method, pl, m_rmse, sd_rmse, m_mae, sd_mae, m_sm, sd_sm, logf)
        } else {
          printf("%s,%s,%s,%d,%.10g,%.10g,%.10g,%.10g,,,%s\n",
                 data, model, method, pl, m_rmse, sd_rmse, m_mae, sd_mae, logf)
        }
      } else {
        printf("%s,%s,%s,%d,,,,,,, %s\n", data, model, method, pl, logf)
      }
    }' "$logfile" >> "$out_csv"
done

echo "Wrote $out_csv"

