import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
import seaborn as sns
from tqdm import tqdm

@dataclass
class SimulationParams:
    n_healthy: int = 100
    n_cancer: int = 5
    max_edges: int = 5
    max_rounds: int = 1000
    cells_per_round: int = 10
    treatment_interval: int = 5
    healthy_kill_prob: float = 0.1
    cancer_kill_prob: float = 0.3
    healthy_spread_threshold: float = 0.6  
    cancer_spread_threshold: float = 0.8   
    fixation_threshold: float = 0.99
    seed: int = 42

class TissueSimulation:
    def __init__(self, params: SimulationParams, treatment_type: str = 'standard'):
        if treatment_type not in ['none', 'standard', 'alternative']:
            raise ValueError("Treatment type must be 'none', 'standard', or 'alternative'")
        self.params = params
        self.treatment_type = treatment_type
        self.history = []
        np.random.seed(self.params.seed)
        self.reset()

    def reset(self):
        total_cells = self.params.n_healthy + self.params.n_cancer
        self.G = nx.Graph()
        self.G.add_nodes_from(range(total_cells))
        self.round = 0
        self.metrics = []
        self._setup_initial_network()
        self.save_state()

    def _setup_initial_network(self):
        total_cells = self.params.n_healthy + self.params.n_cancer
        
        cell_types = {
            i: ('healthy' if i < self.params.n_healthy else 'cancer')
            for i in range(total_cells)
        }
        nx.set_node_attributes(self.G, cell_types, 'type')
        nx.set_node_attributes(self.G, False, 'under_treatment')
        
        for i in range(total_cells):
            possible_neighbors = list(set(range(total_cells)) - {i} - set(self.G.neighbors(i)))
            n_edges_to_add = min(
                self.params.max_edges - self.G.degree(i),
                len(possible_neighbors)
            )
            if n_edges_to_add > 0:
                new_neighbors = np.random.choice(
                    possible_neighbors,
                    size=n_edges_to_add,
                    replace=False
                )
                self.G.add_edges_from((i, n) for n in new_neighbors)

    def get_cell_proportions(self):
        counts = Counter(nx.get_node_attributes(self.G, 'type').values())
        total_living = counts.get('healthy', 0) + counts.get('cancer', 0)
        
        if total_living == 0:
            return 0, 0, 1  # All dead
            
        return (
            counts.get('healthy', 0) / len(self.G),
            counts.get('cancer', 0) / len(self.G),
            counts.get('dead', 0) / len(self.G)
        )

    def check_fixation(self):
        healthy_prop, cancer_prop, _ = self.get_cell_proportions()
        return (healthy_prop >= self.params.fixation_threshold or 
                cancer_prop >= self.params.fixation_threshold or 
                healthy_prop + cancer_prop == 0)

    def save_state(self):
        state = {
            node: {
                'type': data['type'],
                'under_treatment': data['under_treatment']
            }
            for node, data in self.G.nodes(data=True)
        }
        self.history.append(state)

    def apply_treatment(self):
        if self.treatment_type == 'none':
            return
        elif self.treatment_type == 'standard':
            self._standard_treatment()
        else:
            nx.set_node_attributes(self.G, True, 'under_treatment')

    def _standard_treatment(self):
        updates = {}
        for node, data in self.G.nodes(data=True):
            if data['type'] == 'dead':
                continue
            
            kill_prob = (self.params.healthy_kill_prob 
                        if data['type'] == 'healthy' 
                        else self.params.cancer_kill_prob)
            
            if np.random.random() < kill_prob:
                updates[node] = {'type': 'dead'}
        
        nx.set_node_attributes(self.G, updates)

    def simulate_round(self):
      if self.treatment_type != 'none':
          if self.round % self.params.treatment_interval == 0 and self.round > 0:
              self.apply_treatment()

      living_cells = [
          node for node, data in self.G.nodes(data=True)
          if data['type'] != 'dead'
      ]
      
      if len(living_cells) < 2:
          return False
          
      n_select = min(self.params.cells_per_round, len(living_cells))
      selected_cells = np.random.choice(living_cells, size=n_select, replace=False)
      
      updates = {}
      for cell in selected_cells:
          cell_type = self.G.nodes[cell]['type']
          
          if (self.treatment_type != 'none' and 
              self.treatment_type == 'alternative' and 
              self.G.nodes[cell]['under_treatment']):
              kill_prob = (self.params.healthy_kill_prob 
                        if cell_type == 'healthy'
                        else self.params.cancer_kill_prob)
              if np.random.random() < kill_prob:
                  updates[cell] = {'type': 'dead'}
                  continue

          neighbors = list(self.G.neighbors(cell))
          if neighbors:
              target = np.random.choice(neighbors)
              # Use different spread thresholds based on cell type
              spread_threshold = (self.params.cancer_spread_threshold 
                                if cell_type == 'cancer' 
                                else self.params.healthy_spread_threshold)
              
              # Key fix: Only attempt spread if target is not the same type
              # AND invert the threshold comparison for cancer cells
              if self.G.nodes[target]['type'] != cell_type:
                  if cell_type == 'cancer':
                      # For cancer cells, LOWER threshold means MORE spreading
                      if np.random.random() <= spread_threshold:  # Changed < to <=
                          updates[target] = {'type': cell_type}
                  else:
                      # For healthy cells, HIGHER threshold means MORE spreading
                      if np.random.random() <= spread_threshold:  # Changed < to <=
                          updates[target] = {'type': cell_type}

      nx.set_node_attributes(self.G, updates)
      
      if self.treatment_type == 'alternative':
          nx.set_node_attributes(self.G, False, 'under_treatment')

      self._record_metrics()
      self.save_state()
      self.round += 1

      return (not self.check_fixation() and 
              self.round < self.params.max_rounds)
            
    def _record_metrics(self):
        healthy_prop, cancer_prop, dead_prop = self.get_cell_proportions()
        self.metrics.append({
            'round': self.round,
            'healthy_proportion': healthy_prop,
            'cancer_proportion': cancer_prop,
            'dead_proportion': dead_prop,
            'treatment_type': self.treatment_type
        })

    def run_simulation(self, store_history: bool = True) -> pd.DataFrame:
        """
        Run the simulation until fixation or max rounds reached.
        
        Args:
            store_history (bool): Whether to store the state history for animation
            
        Returns:
            pd.DataFrame: DataFrame containing the simulation metrics
        """
        if not store_history:
            self.history = []
        
        while self.simulate_round():
            if not store_history:
                self.history = []
                
        return pd.DataFrame(self.metrics)
    
    def create_animation(self, filename: str, max_frames: int = 50):
        frame_interval = max(1, len(self.history) // max_frames)
        selected_frames = self.history[::frame_interval]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = nx.spring_layout(self.G, k=1, iterations=50)

        color_map = {
            'dead': 'gray',
            'healthy': {'normal': 'green', 'treatment': 'lightgreen'},
            'cancer': {'normal': 'red', 'treatment': 'pink'}
        }

        def update(frame):
            ax.clear()
            state = selected_frames[frame]
            
            node_colors = [
                color_map[state[node]['type']]['treatment' if state[node]['under_treatment'] else 'normal']
                if state[node]['type'] != 'dead' else color_map['dead']
                for node in self.G.nodes()
            ]

            nx.draw(self.G, pos=pos, node_color=node_colors, 
                   with_labels=False, ax=ax, node_size=300)
            ax.set_title(f'Round {frame * frame_interval}')
            return ax,

        ani = animation.FuncAnimation(
            fig, update, frames=len(selected_frames), 
            interval=200, blit=False
        )
        ani.save(filename, writer='ffmpeg')
        plt.close()

def run_parameter_sweep(n_trials: int = 10, base_seed: int = 42) -> pd.DataFrame:
    param_variations = {
        'base': SimulationParams(seed=base_seed),
        'high_virulence': SimulationParams(
            cancer_spread_threshold=0.2,
            healthy_spread_threshold=0.6,
            seed=base_seed
        ),
        'low_virulence': SimulationParams(
            cancer_spread_threshold=0.5,
            healthy_spread_threshold=0.6,
            seed=base_seed
        ),
        'aggressive_treatment': SimulationParams(
            healthy_kill_prob=0.05,
            cancer_kill_prob=0.5,
            seed=base_seed
        ),
        'frequent_treatment': SimulationParams(treatment_interval=3, seed=base_seed),
        'infrequent_treatment': SimulationParams(treatment_interval=10, seed=base_seed),
    }
    
    results = []
    
    for param_name, params in tqdm(param_variations.items(), desc="Parameter sets"):
        for treatment_type in ['none', 'standard', 'alternative']:
            for trial in range(n_trials):
                params.seed = base_seed + trial
                sim = TissueSimulation(params, treatment_type)
                metrics_df = sim.run_simulation(store_history=False)
                
                final_metrics = metrics_df.iloc[-1].to_dict()
                max_cancer_prop = metrics_df['cancer_proportion'].max()
                survival_time = len(metrics_df)
                
                results.append({
                    'parameter_set': param_name,
                    'treatment_type': treatment_type,
                    'trial': trial,
                    'final_healthy_prop': final_metrics['healthy_proportion'],
                    'final_cancer_prop': final_metrics['cancer_proportion'],
                    'final_dead_prop': final_metrics['dead_proportion'],
                    'max_cancer_prop': max_cancer_prop,
                    'survival_time': survival_time,
                    'seed': params.seed,
                    'cancer_spread_threshold': params.cancer_spread_threshold,
                    'healthy_spread_threshold': params.healthy_spread_threshold,
                    **params.__dict__
                })
    
    return pd.DataFrame(results)

def create_analysis_plots(sweep_results: pd.DataFrame):
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Box plot of final cancer proportion
    ax1 = plt.subplot(2, 2, 1)
    sns.boxplot(data=sweep_results, x='parameter_set', y='final_cancer_prop',
                hue='treatment_type', ax=ax1)
    ax1.set_title('Final Cancer Cell Proportion Distribution')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # 2. Box plot of survival time
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(data=sweep_results, x='parameter_set', y='survival_time',
                hue='treatment_type', ax=ax2)
    ax2.set_title('Simulation Survival Time Distribution')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Scatter plot of final proportions
    ax3 = plt.subplot(2, 2, 3)
    for treatment in sweep_results['treatment_type'].unique():
        data = sweep_results[sweep_results['treatment_type'] == treatment]
        ax3.scatter(data['final_healthy_prop'], data['final_cancer_prop'], 
                   label=treatment, alpha=0.6)
    ax3.set_xlabel('Final Healthy Cell Proportion')
    ax3.set_ylabel('Final Cancer Cell Proportion')
    ax3.set_title('Final Cell Type Proportions')
    ax3.legend()
    
    # 4. Bar plot of average max cancer proportion
    ax4 = plt.subplot(2, 2, 4)
    pivot_data = sweep_results.pivot_table(
        values='max_cancer_prop',
        index='parameter_set',
        columns='treatment_type',
        aggfunc='mean'
    )
    pivot_data.plot(kind='bar', ax=ax4)
    ax4.set_title('Average Maximum Cancer Cell Proportion')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_single_run_animations(base_seed: int = 42):
    params = SimulationParams(seed=base_seed)
    for treatment_type in ['none', 'standard', 'alternative']:
        print(f"Creating animation for {treatment_type} treatment...")
        sim = TissueSimulation(params, treatment_type)
        sim.run_simulation(store_history=True)
        sim.create_animation(f'{treatment_type}_treatment.mp4')

def create_additional_plots(sweep_results: pd.DataFrame):
    # 1. Time series plot of cell proportions for each treatment type
    fig = plt.figure(figsize=(20, 15))
    
    # Reconstruct time series data
    time_series_data = []
    for _, row in sweep_results.iterrows():
        param_set = row['parameter_set']
        treatment = row['treatment_type']
        sim = TissueSimulation(
            SimulationParams(**{k: row[k] for k in SimulationParams().__dict__.keys()}),
            treatment
        )
        metrics_df = sim.run_simulation(store_history=False)
        metrics_df['parameter_set'] = param_set
        metrics_df['treatment_type'] = treatment
        metrics_df['trial'] = row['trial']
        time_series_data.append(metrics_df)
    
    time_series_df = pd.concat(time_series_data)
    
    # 1. Average cell proportions over time for each treatment
    ax1 = plt.subplot(2, 2, 1)
    for treatment in ['none', 'standard', 'alternative']:
        treatment_data = time_series_df[time_series_df['treatment_type'] == treatment]
        avg_data = treatment_data.groupby('round').agg({
            'healthy_proportion': 'mean',
            'cancer_proportion': 'mean',
            'dead_proportion': 'mean'
        })
        
        ax1.plot(avg_data.index, avg_data['healthy_proportion'], 
                label=f'{treatment} - Healthy', linestyle='-')
        ax1.plot(avg_data.index, avg_data['cancer_proportion'],
                label=f'{treatment} - Cancer', linestyle='--')
        ax1.plot(avg_data.index, avg_data['dead_proportion'],
                label=f'{treatment} - Dead', linestyle=':')
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cell Proportion')
    ax1.set_title('Average Cell Proportions Over Time by Treatment')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Treatment timing impact visualization
    ax2 = plt.subplot(2, 2, 2)
    treatment_timings = time_series_df[
        time_series_df['parameter_set'].isin(['frequent_treatment', 'infrequent_treatment'])
    ]
    
    for param_set in ['frequent_treatment', 'infrequent_treatment']:
        param_data = treatment_timings[treatment_timings['parameter_set'] == param_set]
        avg_data = param_data.groupby(['round', 'treatment_type']).agg({
            'cancer_proportion': 'mean'
        }).reset_index()
        
        for treatment in ['standard', 'alternative']:
            treatment_avg = avg_data[avg_data['treatment_type'] == treatment]
            ax2.plot(treatment_avg['round'], treatment_avg['cancer_proportion'],
                    label=f'{param_set} - {treatment}')
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cancer Proportion')
    ax2.set_title('Impact of Treatment Timing on Cancer Growth')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Virulence comparison
    ax3 = plt.subplot(2, 2, 3)
    virulence_data = time_series_df[
        time_series_df['parameter_set'].isin(['high_virulence', 'low_virulence', 'base'])
    ]
    
    for param_set in ['high_virulence', 'low_virulence', 'base']:
        param_data = virulence_data[virulence_data['parameter_set'] == param_set]
        avg_data = param_data.groupby('round').agg({
            'cancer_proportion': ['mean', 'std']
        }).reset_index()
        
        mean = avg_data[('cancer_proportion', 'mean')]
        std = avg_data[('cancer_proportion', 'std')]
        
        ax3.plot(avg_data['round'], mean, label=param_set)
        ax3.fill_between(avg_data['round'], 
                        mean - std, 
                        mean + std,
                        alpha=0.2)
    
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cancer Proportion')
    ax3.set_title('Cancer Growth by Virulence Level')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Treatment effectiveness heatmap
    ax4 = plt.subplot(2, 2, 4)
    effectiveness_data = sweep_results.pivot_table(
        values='max_cancer_prop',
        index='cancer_kill_prob',
        columns='healthy_kill_prob',
        aggfunc='mean'
    )
    
    sns.heatmap(effectiveness_data, 
                ax=ax4,
                cmap='RdYlBu_r',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Max Cancer Proportion'})
    
    ax4.set_title('Treatment Effectiveness\nby Kill Probabilities')
    ax4.set_xlabel('Healthy Cell Kill Probability')
    ax4.set_ylabel('Cancer Cell Kill Probability')
    
    plt.tight_layout()
    plt.savefig('additional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    base_seed = 42
    
    print("Creating single-run animations...")
    create_single_run_animations(base_seed)
    
    print("\nRunning parameter sweep with multiple trials...")
    sweep_results = run_parameter_sweep(n_trials=10, base_seed=base_seed)
    
    sweep_results.to_csv('parameter_sweep_results.csv', index=False)
    
    print("\nCreating additional analysis plots...")
    create_additional_plots(sweep_results)

    print("\nCreating analysis plots...")
    create_analysis_plots(sweep_results)

    print("\nSummary Statistics:")
    summary = sweep_results.groupby(['parameter_set', 'treatment_type']).agg({
        'final_cancer_prop': ['mean', 'std'],
        'survival_time': ['mean', 'std'],
        'max_cancer_prop': ['mean', 'std']
    }).round(2)
    
    print(summary)
    
    with open('simulation_summary.txt', 'w') as f:
        f.write("Simulation Summary\n")
        f.write("=================\n\n")
        f.write(str(summary))

if __name__ == "__main__":
    main()