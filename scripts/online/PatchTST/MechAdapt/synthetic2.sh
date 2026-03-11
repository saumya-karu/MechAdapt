#!/usr/bin/env bash
set -euo pipefail


mkdir -p logs/online
mkdir -p results

seq_len=94
model_name="PatchTST"
online_method="MechAdapt"
tune_mode="down_up"
feature="MS"
target_variable="Y"


datasets=(synthetic2_200_30)

mids=(50)
btls=(48)
pred_lens=(24 48 96)
learning_rates=(0.003)
online_learning_rates=(0.00003)


for data in "${datasets[@]}"; do
  echo "=============================="
  echo "Dataset: $data"
  echo "=============================="

  for mid in "${mids[@]}"; do
    for btl in "${btls[@]}"; do
      for pred_len in "${pred_lens[@]}"; do
        for lr in "${learning_rates[@]}"; do
          for olr in "${online_learning_rates[@]}"; do

            suffix="_lr${lr}_onlinelr${olr}"
            logfile="logs/online/${model_name}_${online_method}_mid${mid}_share_fulltune_${tune_mode}_${data}_${pred_len}_btl${btl}${suffix}.log"
            checkpoint_name="${data}_${model_name}_None_sl94_pl${pred_len}"

            echo "Running: $logfile"

            python -u run.py \
              --dataset "$data" --border_type online  --batch_size 128 \
              --features "$feature" \
              --target "$target_variable" \
              --model "$model_name" \
              --seq_len "$seq_len" \
              --pred_len "$pred_len" \
              --tune_mode "$tune_mode" \
              --concept_dim "$mid" \
              --bottleneck_dim "$btl" \
              --val_online_lr --diff_online_lr \
              --itr 3 \
              --save_opt \
              --online_method "$online_method" \
              --target_variate 3 \
              --only_test \
              --online_learning_rate "$olr" \
              --learning_rate "$lr" \
              --lradj type3 \
              > "$logfile" 2>&1

          done
        done
      done
    done
  done
done

echo "All runs finished."


for data in "${datasets[@]}"; do
  out_csv="results/${data}/metrics_${data}_${model_name}_${online_method}.csv"
  echo "data,model,online_method,tune_mode,mid,btl,pred_len,lr,onlinelr,rmse_mean,rmse_std,mae_mean,mae_std,smape_mean,smape_std,log_file" > "$out_csv"

  for mid in "${mids[@]}"; do
    for btl in "${btls[@]}"; do
      for pred_len in "${pred_lens[@]}"; do
        for lr in "${learning_rates[@]}"; do
          for olr in "${online_learning_rates[@]}"; do

            suffix="_lr${lr}_onlinelr${olr}"
            logfile="logs/online/${model_name}_${online_method}_mid${mid}_share_fulltune_${tune_mode}_${data}_${pred_len}_btl${btl}${suffix}.log"

            if [[ ! -f "$logfile" ]]; then
              echo "${data},${model_name},${online_method},${tune_mode},${mid},${btl},${pred_len},${lr},${olr},,,,,${logfile}" >> "$out_csv"
              continue
            fi

            awk -v data="$data" -v model="$model_name" -v method="$online_method" -v tune="$tune_mode" \
              -v mid="$mid" -v btl="$btl" -v pl="$pred_len" -v lr="$lr" -v olr="$olr" -v logf="$logfile" '
            BEGIN{
              n_rmse=0; s_rmse=0; ss_rmse=0;
              n_mae=0;  s_mae=0;  ss_mae=0;
              n_smape=0; s_smape=0; ss_smape=0;
              saw_rmse=0;
            }

            # Case 1: rmse + mae (+ optional smape) on same line
            $0 ~ /rmse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
              saw_rmse=1

              match($0,/rmse:[[:space:]]*([0-9.eE+\-]+)/,r); rm=r[1]+0
              match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a);  ma=a[1]+0

              s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++
              s_mae+=ma;  ss_mae+=ma*ma;  n_mae++

              # try to read smape if present on this line (case-insensitive, optional %)
              if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
                sp=s[1]+0
                s_smape+=sp; ss_smape+=sp*sp; n_smape++
              }
              next
            }

            # Case 2: mse + mae (+ optional smape) on same line (rmse = sqrt(mse))
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

            # Case 3: smape reported alone on a separate line (SMAPE: 12.3 or smape: 0.123, etc.)
            # We only collect SMAPE if we have already started seeing metrics in the log (optional guard),
            # but this still works even if smape appears earlier.
            tolower($0) ~ /smape:[[:space:]]*[0-9.eE+\-]+/ {
              match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)
              sp=s[1]+0
              s_smape+=sp; ss_smape+=sp*sp; n_smape++
              next
            }

            END{
              have_rmse = (n_rmse>0)
              have_mae  = (n_mae>0)
              have_smape = (n_smape>0)

              if (have_rmse && have_mae) {
                m_rmse=s_rmse/n_rmse
                v_rmse=ss_rmse/n_rmse - m_rmse*m_rmse; if (v_rmse<0) v_rmse=0
                sd_rmse=sqrt(v_rmse)

                m_mae=s_mae/n_mae
                v_mae=ss_mae/n_mae - m_mae*m_mae; if (v_mae<0) v_mae=0
                sd_mae=sqrt(v_mae)

                if (have_smape) {
                  m_smape=s_smape/n_smape
                  v_smape=ss_smape/n_smape - m_smape*m_smape; if (v_smape<0) v_smape=0
                  sd_smape=sqrt(v_smape)
                  printf("%s,%s,%s,%s,%d,%d,%d,%s,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n",
                         data, model, method, tune, mid, btl, pl, lr, olr,
                         m_rmse, sd_rmse, m_mae, sd_mae, m_smape, sd_smape, logf)
                } else {
                  printf("%s,%s,%s,%s,%d,%d,%d,%s,%s,%.10g,%.10g,%.10g,%.10g,,,%s\n",
                         data, model, method, tune, mid, btl, pl, lr, olr,
                         m_rmse, sd_rmse, m_mae, sd_mae, logf)
                }

              } else {
                printf("%s,%s,%s,%s,%d,%d,%d,%s,%s,,,,,,,%s\n",
                       data, model, method, tune, mid, btl, pl, lr, olr, logf)
              }
            }' "$logfile" >> "$out_csv"


          done
        done
      done
    done
  done
done

echo "Done. Metrics written to $out_csv"
