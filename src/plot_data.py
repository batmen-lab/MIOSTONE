import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import Pipeline


class PlotDataPipeline(Pipeline):
    def _create_output_subdir(self):
        pass

    def _plot_taxonomic_composition(self):
        original_taxonomic_composition = [[] for _ in range(self.tree.max_depth + 1)]
        for node, depth in self.tree.depths.items():
            original_taxonomic_composition[depth].append(node)

        pruned_taxonomic_composition = [[] for _ in range(self.tree.max_depth + 1)]
        for node in self.tree.ete_tree.traverse():
            pruned_taxonomic_composition[self.tree.depths[node.name]].append(node.name)

        # Save the taxonomic composition
        taxonomic_composition_dict = {
            f'{self.tree.taxonomic_ranks[i]}': pruned_taxonomic_composition[i] for i in range(1, self.tree.max_depth + 1)
        }
        with open(f'../data/{self.data_target}.json', 'w') as f:
            json.dump(taxonomic_composition_dict, f)

        # Plot the taxonomic composition
        plt.figure(figsize=(15, 15))
        sns.set_theme(style='white')

        num_original_taxonomic_composition = [len(taxa) for taxa in original_taxonomic_composition[1:]]
        num_pruned_taxonomic_composition = [len(taxa) for taxa in pruned_taxonomic_composition[1:]]

        ax = sns.barplot(x=self.tree.taxonomic_ranks[1:],
                    y=[pruned / original for pruned, original in zip(num_pruned_taxonomic_composition, num_original_taxonomic_composition)],
                    hue=self.tree.taxonomic_ranks[1:],
                    dodge=False)
        
        # Remove the legend
        ax.get_legend().remove()

        # Anotate the bars with the pruned / original 
        for i, v in enumerate([pruned / original for pruned, original in zip(num_pruned_taxonomic_composition, num_original_taxonomic_composition)]):
            pruned = num_pruned_taxonomic_composition[i]
            original = num_original_taxonomic_composition[i]
            ax.text(i, v + 0.01, 
                    f'{pruned} / {original}',
                    ha='center', 
                    va='bottom')
            
        plt.xlabel('')
        plt.ylabel('Proportion (Pruned / Original)')
        plt.title(self.data_target)
       
        plt.show()

    def run(self, dataset, target, *args, **kwargs):
        # Define filepaths
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk',
        }
        self.data_target = f'{dataset}_{target}'
        self._load_tree(self.filepaths['tree'])
        self.tree.compute_depths()
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'], preprocess=False)
        self._plot_taxonomic_composition()
    
def run_plot_data_pipeline(dataset, target, *args, **kwargs):
    pipeline = PlotDataPipeline(seed=0)
    pipeline.run(dataset, target, *args, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--target', type=str, help='Target name')
    args = parser.parse_args() 

    run_plot_data_pipeline(args.dataset, args.target)