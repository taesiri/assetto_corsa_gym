import os
import sys
import time

from omegaconf import OmegaConf

sys.path.append(os.path.abspath("./assetto_corsa_gym"))

import AssettoCorsaEnv.assettoCorsa as assettoCorsa


def main():
    cfg = OmegaConf.load("config.yml")
    cfg.AssettoCorsa.screen_capture_enable = False
    cfg.AssettoCorsa.track = "monza"
    cfg.AssettoCorsa.car = "ks_mazda_miata"

    env = None
    try:
        env = assettoCorsa.make_ac_env(cfg=cfg, work_dir="outputs/smoke_test")
        env.reset()

        for step in range(300):
            action = env.action_space.sample()
            env.set_actions(action)
            _, reward, done, _ = env.step(action=None)
            if step % 50 == 0:
                print(f"t={step} reward={reward} done={done}")
            if done:
                break
            time.sleep(0.01)

        env.recover_car()
        print("OK: smoke test finished")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
