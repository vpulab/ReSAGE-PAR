
#rethinking baseline resnet50
#mlflow run . --experiment-name=rapzs --run-name=resnet50_syn_x2_with-1s --env-manager=local \
#  -P cfg=./configs/pedes_baseline/rap_zsgithub_baseline_Synx2_with-1s.yaml

#mlflow run . --experiment-name=rapv2 --run-name=resnet50_syn_x2_with-1s --env-manager=local \
#    -P cfg=./configs/pedes_baseline/rap_v2github_baseline_Synx2_with-1s.yaml

#mlflow run . --experiment-name=rapv1 --run-name=resnet50_syn_x2_with-1s --env-manager=local \
#    -P cfg=./configs/pedes_baseline/rap_v1github_baseline_Synx2_with-1s.yaml

#mlflow run . --experiment-name=petazs --run-name=resnet50_syn_x2_with-1s --env-manager=local \
#    -P cfg=./configs/pedes_baseline/peta_zsgithub_baseline_Synx2_with-1s.yaml

#mlflow run . --experiment-name=peta --run-name=resnet50_syn_x2_with-1s --env-manager=local \
#    -P cfg=./configs/pedes_baseline/peta_github_baseline_Synx2_with-1s.yaml

#mlflow run . --experiment-name=pa100k --run-name=resnet50_syn_x2_with-1s --env-manager=local \
#    -P cfg=./configs/pedes_baseline/pa100k_github_baseline_Synx2_with-1s.yaml


# binception with -1s
#mlflow run . --experiment-name=rapzs --run-name=bninception_syn_x2_with-1s --env-manager=local \
#  -P cfg=./configs/pedes_baseline/rap_zsgithub_baseline_Synx2_BIN_with-1s.yaml

mlflow run . --experiment-name=petazs --run-name=bninception_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/peta_zsgithub_baseline_Synx2_BIN_with-1s.yaml

mlflow run . --experiment-name=peta --run-name=bninception_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/peta_github_baseline_Synx2_BIN_with-1s.yaml

mlflow run . --experiment-name=rapv1 --run-name=bninception_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/rap_v1github_baseline_Synx2_BIN_with-1s.yaml

mlflow run . --experiment-name=rapv2 --run-name=bninception_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/rap_v2github_baseline_Synx2_BIN_with-1s.yaml

mlflow run . --experiment-name=pa100k --run-name=bninception_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/pa100k_github_baseline_Synx2_BIN_with-1s.yaml


# swin with -1s
mlflow run . --experiment-name=rapzs --run-name=swin_syn_x2_with-1s --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_zsgithub_baseline_Synx2_Swin_with-1s.yaml

mlflow run . --experiment-name=petazs --run-name=swin_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/peta_zsgithub_baseline_Synx2_Swin_with-1s.yaml

mlflow run . --experiment-name=peta --run-name=swin_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/peta_github_baseline_Synx2_Swin_with-1s.yaml

mlflow run . --experiment-name=rapv1 --run-name=swin_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/rap_v1github_baseline_Synx2_Swin_with-1s.yaml

mlflow run . --experiment-name=rapv2 --run-name=swin_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/rap_v2github_baseline_Synx2_Swin_with-1s.yaml

mlflow run . --experiment-name=pa100k --run-name=swin_syn_x2_with-1s --env-manager=local \
    -P cfg=./configs/pedes_baseline/pa100k_github_baseline_Synx2_Swin_with-1s.yaml
