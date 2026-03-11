if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=336
data=ETTh2
datasets=("$data")
model_name=PatchTST
online_method=SOLID
target_variable=OT
feature=MS
learning_rate=0.0001
online_learning_rate=0.00001

test_train_nums=(500)
selected_data_nums=(5)
pred_lens=(24 48 96)
online_learning_rates=(0.00001)

for test_train_num in 500
do
for selected_data_num in 5
do
for pred_len in 24 48 96
do
for online_learning_rate in 0.00001
do
  filename=logs/online/$model_name'_'$online_method'whole_cont_'$data'_'$pred_len'_N'$test_train_num'_n'$selected_data_num'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --features $feature \
    --target $target_variable \
    --target_variate 6\
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method --whole_model --continual \
    --only_test --test_train_num $test_train_num --selected_data_num $selected_data_num \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
done
done
mkdir -p results
pred_lens=(24 48 96)
for data in "${datasets[@]}"; do
  mkdir -p "results/${data}"
  out_csv="results/${data}/metrics_${data}_${model_name}_${online_method}.csv"
  echo "data,model,online_method,pred_len,rmse_mean,rmse_std,mae_mean,mae_std,smape_mean,smape_std,log_file" > "$out_csv"

  for test_train_num in 500; do
    for selected_data_num in 5; do
      for pred_len in "${pred_lens[@]}"; do
        for online_learning_rate in 0.00001; do

          logfile=logs/online/$model_name'_'$online_method'whole_cont_'$data'_'$pred_len'_N'$test_train_num'_n'$selected_data_num'_onlinelr'$online_learning_rate.log
          if [[ ! -f "$logfile" ]]; then
            echo "warn: missing $logfile"
            echo "${data},${model_name},${online_method},${pred_len},,,,,,,${logfile}" >> "$out_csv"
            continue
          fi

         awk -v data="$data" -v model="$model_name" -v method="$online_method" \
    -v pl="$pred_len" -v logf="$logfile" '
BEGIN{
  n_rmse=0; s_rmse=0; ss_rmse=0;
  n_mae=0;  s_mae=0;  ss_mae=0;
  n_sm=0;   s_sm=0;   ss_sm=0;
  saw_rmse=0;
}

# Case 1: "rmse: X, mae: Y" (and maybe smape on same line)
$0 ~ /rmse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
  saw_rmse=1;

  match($0,/rmse:[[:space:]]*([0-9.eE+\-]+)/,r); rm=r[1]+0;
  match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a);  ma=a[1]+0;

  s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++;
  s_mae +=ma; ss_mae +=ma*ma; n_mae++;

  if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
    sm=s[1]+0;
    s_sm +=sm; ss_sm +=sm*sm; n_sm++;
  }
  next;
}

# Case 2: "mse: X, mae: Y" => rmse=sqrt(mse)
(!saw_rmse) && $0 ~ /mse:[[:space:]]*[0-9.eE+\-]+,[[:space:]]*mae:[[:space:]]*[0-9.eE+\-]+/ {
  match($0,/mse:[[:space:]]*([0-9.eE+\-]+)/,m); ms=m[1]+0;
  match($0,/mae:[[:space:]]*([0-9.eE+\-]+)/,a); ma=a[1]+0;

  rm=sqrt(ms);
  s_rmse+=rm; ss_rmse+=rm*rm; n_rmse++;
  s_mae +=ma; ss_mae +=ma*ma; n_mae++;

  if (match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s)) {
    sm=s[1]+0;
    s_sm +=sm; ss_sm +=sm*sm; n_sm++;
  }
  next;
}

# Case 3: smape alone on its own line
tolower($0) ~ /smape:[[:space:]]*[0-9.eE+\-]+/ {
  match(tolower($0),/smape:[[:space:]]*([0-9.eE+\-]+)/,s);
  sm=s[1]+0;
  s_sm +=sm; ss_sm +=sm*sm; n_sm++;
  next;
}

END{
  if (n_rmse>0 && n_mae>0) {
    m_rmse=s_rmse/n_rmse;
    v_rmse=ss_rmse/n_rmse - m_rmse*m_rmse; if (v_rmse<0) v_rmse=0;
    sd_rmse=sqrt(v_rmse);

    m_mae=s_mae/n_mae;
    v_mae=ss_mae/n_mae - m_mae*m_mae; if (v_mae<0) v_mae=0;
    sd_mae=sqrt(v_mae);

    if (n_sm>0) {
      m_sm=s_sm/n_sm;
      v_sm=ss_sm/n_sm - m_sm*m_sm; if (v_sm<0) v_sm=0;
      sd_sm=sqrt(v_sm);

      printf("%s,%s,%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n",
             data, model, method, pl,
             m_rmse, sd_rmse, m_mae, sd_mae, m_sm, sd_sm, logf);
    } else {
      printf("%s,%s,%s,%d,%.10g,%.10g,%.10g,%.10g,,,%s\n",
             data, model, method, pl,
             m_rmse, sd_rmse, m_mae, sd_mae, logf);
    }
  } else {
    # No metrics found in logfile
    printf("%s,%s,%s,%d,,,,,,,%s\n", data, model, method, pl, logf);
  }
}
' "$logfile" >> "$out_csv"

        done
      done
    done
  done

  echo "Wrote $out_csv"
done
