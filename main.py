from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from Multi_UAV_env import Multi_UAV_env_multi_level_practical_multicheck

if __name__ == '__main__':
    idx = 0
    print(idx)
    args = get_common_args()
    args = get_mixer_args(args)
    num_v_level = 2
    env = Multi_UAV_env_multi_level_practical_multicheck(num_v_level)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.n_channel = env_info["n_channel"]
    args.num_scene = 10
    runner = Runner(env, args)
    if not args.evaluate:
        runner.run(idx)
    else:
        win_rate, _ = runner.evaluate()
        print('The win rate of {} is  {}'.format(args.alg, win_rate))