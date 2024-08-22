import hydra

from sd3d.bbox_prediction.policies import BboxPredictorLMPolicy


@hydra.main(version_base=None, config_path="/home/mila/a/anthony.gosselin/dev/Ctrl-V/src/sd3d/bbox_prediction/cfgs/", config_name="config")
def main(cfg):
    policy = BboxPredictorLMPolicy(cfg)
    policy.run(cfg.num_eval_samples)


if __name__ == '__main__':
    main()