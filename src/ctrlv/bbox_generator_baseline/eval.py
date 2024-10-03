import hydra

from ctrlv.bbox_generator_baseline.policies import BboxPredictorLMPolicy


@hydra.main(version_base=None, config_path="/home/mila/a/anthony.gosselin/dev/Ctrl-V_dev/src/ctrlv/bbox_prediction/cfgs/", config_name="config")
def main(cfg):
    policy = BboxPredictorLMPolicy(cfg)
    policy.run(cfg.num_eval_samples)


if __name__ == '__main__':
    main()