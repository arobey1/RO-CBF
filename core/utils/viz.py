import seaborn as sns
import pandas as pd
import os 
import matplotlib.pyplot as plt
import wandb

class Visualizer:
    def __init__(self, net, results_path, data_dict, cbf_fn):
        self._net = net
        self._results_path = results_path
        self._data_dict = data_dict
        self._cbf_fn = cbf_fn

        sns.set_style('darkgrid')

    def state_separation(self, params):

        def make_df(safe):
            output = self._net.apply(params, self._data_dict[safe])
            df = pd.DataFrame(output, columns=['h(x)'])
            df['Constraint-Type'] = safe.capitalize()
            return df

        safe_df, unsafe_df = make_df(safe='safe'), make_df(safe='unsafe')

        # cbf_output = self._cbf_fn(params, self._df[self._state_cols].to_numpy(),
        #                             self._df[self._dist_cols].to_numpy())
        # cbf_df = pd.DataFrame(cbf_output, columns=['h(x)'])
        # cbf_df['Constraint-Type'] = 'CBF'

        # df = pd.concat([safe_df, unsafe_df, cbf_df], ignore_index=True)

        df = pd.concat([safe_df, unsafe_df], ignore_index=True)
        
        plt.figure()
        sns.boxplot(data=df, x='Constraint-Type', y='h(x)')
        wandb.log({'separation': wandb.Image(plt)})
        plt.savefig(os.path.join(self._results_path, 'state_separation.png'))
        plt.close()

        # plt.figure()
        # sns.displot(data=df, x='h(x)', col='Constraint-Type')
        # plt.savefig(os.path.join(self._results_path, 'state_distributions.png'))
        # plt.draw()
        # plt.close()