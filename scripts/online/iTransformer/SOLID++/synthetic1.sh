#!/usr/bin/env bash
set -euo pipefail


mkdir -p logs
mkdir -p logs/online


seq_len=94
model_name="iTransformer"
online_method="SOLID"
target_variable=Y
feature="MS"
border_type="online"


datasets=(
  synthetic1_200_30
)


test_train_nums=(500)
selected_data_nums=(5)
pred_lens=(24 48 96)
online_learning_rates=(0.000003)


for data in "${datasets[@]}"; do
  echo "=============================="
  echo "Dataset: $data"
  echo "=============================="

  for test_train_num in "${test_train_nums[@]}"; do
    for selected_data_num in "${selected_data_nums[@]}"; do
      for pred_len in "${pred_lens[@]}"; do
        for online_learning_rate in "${online_learning_rates[@]}"; do

          logfile="logs/online/${model_name}_${online_method}_whole_cont_${data}_${pred_len}_N${test_train_num}_n${selected_data_num}_onlinelr${online_learning_rate}.log"

          echo "Running: $logfile"

          python -u run.py \
            --dataset "$data" \
            --border_type "$border_type" \
            --model "$model_name" \
            --features "$feature" \
            --target "$target_variable" \
            --seq_len "$seq_len" \
            --pred_len "$pred_len" \
            --itr 3 \
            --skip "$logfile" \
            --online_method "$online_method" \
            --whole_model \
            --continual \
            --only_test \
            --target_variate 3 \
            --test_train_num "$test_train_num" \
            --selected_data_num "$selected_data_num" \
            --online_learning_rate "$online_learning_rate" \
            > "$logfile" 2>&1

        done
      done
    done
  done
done

echo "All SOLID++ runs finished."

for data in "${datasets[@]}"; do
  mkdir -p "results/${data}"

  out_csv="results/${data}/metrics_${data}_${model_name}_${online_method}.csv"
  echo "data,model,online_method,pred_len,rmse_mean,rmse_std,mae_mean,mae_std,smape_mean,smape_std,log_file" > "$out_csv"

  for test_train_num in "${test_train_nums[@]}"; do
    for selected_data_num in "${selected_data_nums[@]}"; do
      for pred_len in "${pred_lens[@]}"; do
        for online_learning_rate in "${online_learning_rates[@]}"; do

          logfile="logs/online/${model_name}_${online_method}_whole_cont_${data}_${pred_len}_N${test_train_num}_n${selected_data_num}_onlinelr${online_learning_rate}.log"

          if [[ ! -f "$logfile" ]]; then
            echo "${data},${model_name},${online_method},${pred_len},,,,,,${logfile}" >> "$out_csv"
            continue
          fi

          awk -v data="$data" -v model="$model_name" -v method="$online_method" \
              -v pl="$pred_len" -v logf="$logfile" '
          BEGIN{
            n_rmse=0; s_rmse=0; ss_rmse=0;
            n_mae=0;  s_mae=0;  ss_mae=0;
            n_smape=0; s_smape=0; ss_smape=0;
            saw_rmse=0;
          }

          # rmse + mae on one line
          $0 ~ /rmse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
            saw_rmse=1
            match($0,/rmse:[[:space:]]*([0-9.eE+\-]+)/,r); rm=r[1]+0
            match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a);  ma=a[1]+0
            s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++
            s_mae+=ma;  ss_mae+=ma*ma;  n_mae++
            if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
              sp=s[1]+0
              s_smape+=sp; ss_smape+=sp*sp; n_smape++
            }
            next
          }

          # mse + mae (compute rmse)
          !saw_rmse && $0 ~ /mse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
            match($0,/mse:[[:space:]]*([0-9.eE+\-]+)/,m); ms=m[1]+0
            match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a);  ma=a[1]+0
            rm=sqrt(ms)
            s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++
            s_mae+=ma;  ss_mae+=ma*ma;  n_mae++
            if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
              sp=s[1]+0
              s_smape+=sp; ss_smape+=sp*sp; n_smape++
            }
            next
          }

          # smape on its own line
          tolower($0) ~ /smape:[[:space:]]*[0-9.eE+\-]+/ {
            match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)
            sp=s[1]+0
            s_smape+=sp; ss_smape+=sp*sp; n_smape++
            next
          }

          END{
            if (n_rmse>0 && n_mae>0) {
              m_rmse=s_rmse/n_rmse
              v_rmse=ss_rmse/n_rmse - m_rmse*m_rmse; if (v_rmse<0) v_rmse=0
              sd_rmse=sqrt(v_rmse)

              m_mae=s_mae/n_mae
              v_mae=ss_mae/n_mae - m_mae*m_mae; if (v_mae<0) v_mae=0
              sd_mae=sqrt(v_mae)

              if (n_smape>0) {
                m_smape=s_smape/n_smape
                v_smape=ss_smape/n_smape - m_smape*m_smape; if (v_smape<0) v_smape=0
                sd_smape=sqrt(v_smape)

                printf("%s,%s,%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n",
                       data, model, method, pl,
                       m_rmse, sd_rmse, m_mae, sd_mae, m_smape, sd_smape, logf)
              } else {
                printf("%s,%s,%s,%d,%.10g,%.10g,%.10g,%.10g,,,%s\n",
                       data, model, method, pl,
                       m_rmse, sd_rmse, m_mae, sd_mae, logf)
              }
            } else {
              printf("%s,%s,%s,%d,,,,,,, %s\n", data, model, method, pl, logf)
            }
          }' "$logfile" >> "$out_csv"

        done
      done
    done
  done

  echo "Wrote: $out_csv"
done
