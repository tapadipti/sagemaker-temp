export DVC_REMOTE=s3://tapa-test-s3/dvc-remote/fm-sm

dvc remote add -d storage $DVC_REMOTE

dvc stage add --run -n load_data -d src/load_data.py -o output/data.pkl python src/load_data.py
dvc stage add --run -n train -p train.batch_size,train.hidden_units,train.dropout,train.num_epochs,train.lr -d src/train.py -d output/data.pkl -M dvclive/metrics.json -o output/myfmmodel.keras -o output/myfmmodel.tar.gz --plots-no-cache output/train_logs.csv  --plots-no-cache dvclive/plots/metrics/accuracy.tsv python src/train.py
dvc stage add --run -n evaluate -d src/evaluate.py -d output/data.pkl -d output/myfmmodel.keras -M output/metrics.json --plots-no-cache output/predictions.json --plots-no-cache output/test/samples_of_mispredicted_images/ python src/evaluate.py
dvc plots modify output/predictions.json --template confusion -x actual -y predicted
dvc plots modify output/train_logs.csv --template linear -x epoch -y accuracy


# Get the model version from the dvc file in the Git commit