import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

if __name__ == '__main__':
    def runningMeanAndStd(means, std):
        _num_convo = 100
        _r_mean = means
        _r_std = std
        for j in range(len(means)):
            _r_mean[j] = np.mean(means[0+np.abs(j-_num_convo//2):j+1+_num_convo//2])
            _r_std[j] = np.std(means[0+np.abs(j-_num_convo//2):j+1+_num_convo//2])
        return _r_mean, _r_std

    # SAC-D
    sacd_eval_reward = np.load('SAC/eval_rewards_append_stochastic.npy')
    sacd_eval_reward_mean, sacd_eval_reward_std = runningMeanAndStd(np.max(sacd_eval_reward, axis=0), np.std(sacd_eval_reward, axis=0))
    plt.plot(np.arange(len(sacd_eval_reward_mean)), sacd_eval_reward_mean, color='darkorange', alpha=0.75, label='SAC-S')
    plt.fill_between(np.arange(len(sacd_eval_reward_mean)), 
                     sacd_eval_reward_mean+sacd_eval_reward_std, 
                     sacd_eval_reward_mean-sacd_eval_reward_std, color='darkorange', alpha=0.1)

    # SAC-S
    sacs_eval_reward = np.load('SAC/eval_rewards_append_deterministic.npy')
    sacs_eval_reward_mean, sacs_eval_reward_std = runningMeanAndStd(np.max(sacs_eval_reward, axis=0), np.std(sacs_eval_reward, axis=0))
    plt.plot(np.arange(len(sacs_eval_reward_mean)), sacs_eval_reward_mean, color='black', alpha=0.75, label='SAC-D')
    plt.fill_between(np.arange(len(sacs_eval_reward_mean)), 
                     sacs_eval_reward_mean+sacs_eval_reward_std, 
                     sacs_eval_reward_mean-sacs_eval_reward_std, color='black', alpha=0.1)


    # D-QL (Diffusion-QL)
    dql_eval_reward = np.load('Diffusion_QL/norm_eval_rewards_append.npy')
    dql_eval_reward_mean, dql_eval_reward_std = runningMeanAndStd(np.max(dql_eval_reward, axis=0), np.std(dql_eval_reward, axis=0))
    plt.plot(np.arange(len(dql_eval_reward_mean)), dql_eval_reward_mean, color='green', alpha=0.75, label='D-QL')
    plt.fill_between(np.arange(len(dql_eval_reward_mean)), 
                     dql_eval_reward_mean+dql_eval_reward_std, 
                     dql_eval_reward_mean-dql_eval_reward_std, color='green', alpha=0.1)

    # GPDP-v0
    gpdp_eval_reward = np.load('GPDP/norm_eval_rewards_append.npy')
    gpdp_eval_reward_mean, gpdp_eval_reward_std = runningMeanAndStd(np.max(gpdp_eval_reward, axis=0), np.std(gpdp_eval_reward, axis=0))
    plt.plot(np.arange(len(gpdp_eval_reward_mean)), gpdp_eval_reward_mean, 
                color='blue',
                alpha=1.0, 
                linewidth=2, 
                label='GPDP')
    plt.fill_between(np.arange(len(gpdp_eval_reward_mean)), 
                     gpdp_eval_reward_mean+gpdp_eval_reward_std, 
                     gpdp_eval_reward_mean-gpdp_eval_reward_std, color='blue', alpha=0.1)

    plt.vlines(x=300, ymin=-1, ymax=7.2, colors='red', linestyles='--')
    plt.vlines(x=400, ymin=-1, ymax=7.2, colors='red', linestyles='--')
    plt.ylim(-1, 7.2)
    plt.xlabel('Time step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig('reward_trajectory.png', format='png', dpi=1200)
    plt.tight_layout()
    plt.show()
