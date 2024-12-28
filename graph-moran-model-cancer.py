import networkx as nx
import random
import matplotlib.pyplot as plt

# state in this graph is represented by (# healthy cells, # cancer cells)
# changes in state at each iteration -> # cancer cells can increase by 1, decrease by 1, stay the same

# create complete graph with N nodes (representing N cells)
# edges weighted randomly by distance between cells
# N = total # nodes
def complete_graph(N, num_cancer_cells):

    # complete graph with N nodes
    G = nx.complete_graph(N)

    # assign every node to be healthy except one
    for i in range(N):
        if (i >= num_cancer_cells):
            G.nodes[i]['type'] = 'Healthy'
        else:
            G.nodes[i]['type'] = 'Cancer'
    
    # assign edge weights
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(*(0,5))

    return G

# function to perform one step of the moran model
# G = current graph, N = total # nodes, i = total # cancer cells
# h_b = selective benefit of healthy cells, c_b = selective benefit of cancer cells
def moran_model_step(G, N, i, h_b, c_b):
    # randomly choose a cell for reproduction, based on selective benefit
    # randomly choose neighboring cell for death/replacement
    # number of cancer increases by 1, number of cancer decreases by 1, number stays same
    # first term in formula = prob of choosing individual to reproduce
    # second term = prob of choosing neighboring individual based on proximity to die/replace
    inc_prob = ((i * c_b) / ((i * c_b) + ((N-i) * h_b))) * (N-i)/N
    dec_prob = (((N-i) * h_b) / ((i * c_b) + ((N-i) * h_b))) * i/N
    no_change_prob = 1 - inc_prob - dec_prob

    # pick transition
    chosen_transition = random.choices(["increase", "decrease", "no-change"], [inc_prob, dec_prob, no_change_prob])[0]

    if (chosen_transition == "increase"):
        # increase number of cancer cells by 1 (cancer cell divides and replaces a healthy cell)
        i += 1
        
        # go through all the cancer cells in the graph
        # choose highest likely cancer - healthy pairing (closest distance) to apply transition to

        prob_list = []

        for i in range(N):
            if G.nodes[i]['type'] == 'Cancer':
                neighbors = [n for n in G.neighbors(i) if G.nodes[n]['type'] == 'Healthy']

                # if current cancer cell is not connected to any healthy cells
                if not neighbors:
                    continue

                total_weight = sum((1 / G[i][neighbor]['weight']) for neighbor in neighbors)

                for neighbor in neighbors:
                    replacement_prob = (1 / G[i][neighbor]['weight']) / total_weight
                    prob_list.append(((G.nodes[i], neighbor), replacement_prob))

        if prob_list:
            # find pairing with maximum probability
            cancer_cell, healthy_cell = max(prob_list, key=lambda x: x[1])[0]

            # update graph with chosen pair and based on state change
            G.nodes[healthy_cell]['type'] = 'Cancer'   

    elif (chosen_transition == "decrease"):
        # decrease number of cancer cells by 1 (healthy cell divides and replaces a cancer cell)
        i -= 1

        # get all healthy - cancer pairings
        all_pairings = []

        for i in range(N):
            if G.nodes[i]['type'] == 'Healthy':
                neighbors = [n for n in G.neighbors(i) if G.nodes[n]['type'] == 'Cancer']

                # if current healthy cell is not connected to any cancer cell
                if not neighbors:
                    continue

                for neighbor in neighbors:
                    all_pairings.append((G.nodes[i], neighbor))

        if all_pairings:
            # randomly choose healthy - cancer pairing to apply transition to
            healthy_cell, cancer_cell = random.choice(all_pairings)

            # update graph with chosen pair and based on state change
            G.nodes[cancer_cell]['type'] = 'Healthy'

    return G

def apply_treatment(G, h_b, c_b):
    for node in list(G.nodes): 
        cell_type = G.nodes[node]['type']

        if cell_type == 'Cancer':
            # Probability of killing cancer cell based on its selective benefit
            kill_prob = c_b / (c_b + h_b)
            if random.random() < kill_prob:
                # Kill the cancer cell (change type to 'Dead')
                G.nodes[node]['type'] = 'Dead'

        elif cell_type == 'Healthy':
            # Probability of killing healthy cell based on its selective benefit
            kill_prob = h_b / (c_b + h_b)
            if random.random() < kill_prob:
                # Kill the healthy cell (change type to 'Dead')
                G.nodes[node]['type'] = 'Dead'

    return G

# simulation run for T timesteps
# every j steps, a round of treatment will be applied
def moran_model_simulation(T, j, N, i, h_b, c_b):
    graph = complete_graph(N, i)

    print("GRAPH STATE:")
    for node, data in graph.nodes(data=True):
        print(f"Node {node}: {data}")

    t = 1
    while (t < T):
        print("Timestep t: ", t)
        if (t % j == 0):
            print("APPLY TREATMENT")
            # apply treatment (chemotherapy)
            graph = apply_treatment(graph, h_b, c_b)

        graph = moran_model_step(graph, N, i, h_b, c_b)

        # Calculate and print the number of cancer cells
        num_cancer = 0
        for node, data in graph.nodes(data=True):
            if data['type'] == 'Cancer':
                num_cancer += 1

        # Print the graph state
        print("GRAPH STATE:")
        for node, data in graph.nodes(data=True):
            print(f"Node {node}: {data}")

        print("Number of Cancer Cells: ", num_cancer)

        t += 1

    return graph

healthy_selective_benefit = 1.0
cancer_selective_benefit = 1.5
start_cancer_cell_ct = 10
total_cell_ct = 50
num_simulations = 800
treatment_round_time = num_simulations+10

result_graph = moran_model_simulation(num_simulations, treatment_round_time, 
                                     total_cell_ct, start_cancer_cell_ct, 
                                     healthy_selective_benefit, cancer_selective_benefit)

def visualize_graph(G):
    """
    Visualizes the graph with node types and edge weights.
    - Healthy nodes are blue.
    - Cancer nodes are red.
    """
    pos = nx.spring_layout(G)  # Layout for visualization
    node_colors = ['blue' if G.nodes[node]['type'] == 'healthy' else 'red' for node in G.nodes()]
    
    # Draw nodes with colors based on their type
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_color="white")
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    
    plt.title("Complete Graph with Healthy and Cancer Nodes")
    plt.show()

# Example usage
N = 4  # Total number of nodes

#graph = complete_graph(N)

#visualize_graph(result_graph)
