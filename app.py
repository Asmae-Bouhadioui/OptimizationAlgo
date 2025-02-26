import streamlit as st
import random
import numpy as np
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

# Définition de pb multi-objectifs avec des critères plus riches
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # Maximiser créativité, ergonomie et efficacité
creator.create("Individual", list, fitness=creator.FitnessMulti)

def create_individual():
    return [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]  # [Créativité, Ergonomie, Efficacité]

def evaluate(individual):
    creativity, ergonomics, efficiency = individual
    return creativity, ergonomics, efficiency

# Définition de la toolbox
toolbox = base.Toolbox()
toolbox.register("select", tools.selNSGA2)
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover function
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation function
toolbox.register("evaluate", evaluate)

# Interface Streamlit
st.title("Optimization Algorithms with GAI")
st.sidebar.header("Choose your algorithm")

algorithm_choice = st.sidebar.selectbox("Select the algorithm: ", ["NSGA-II", "Pareto", "AHP", "ACO"])

# Exécution de l'algorithme
def run_nsga2(taille_population, nb_generations):
    pop = toolbox.population(n=taille_population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=taille_population, lambda_=taille_population,
                                              cxpb=0.5, mutpb=0.2, ngen=nb_generations, 
                                              stats=stats, halloffame=hof, verbose=True)
    return pop, logbook

# Pareto Optimization
def run_pareto(taille_population):
    pop = toolbox.population(n=taille_population)
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit  # Assignation correcte des valeurs de fitness
    
    pareto_fronts = tools.sortNondominated(pop, k=taille_population, first_front_only=True)
    return pareto_fronts[0]  # Retourne le premier front de Pareto

# AHP (Analytic Hierarchy Process)
def run_ahp(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    principal_eigvec = eigvecs[:, np.argmax(eigvals)]
    weights = np.real(principal_eigvec / sum(principal_eigvec))  # Normalisation
    return weights

# ACO (Ant Colony Optimization)
def run_aco():
    num_ants = 10
    num_iterations = 50
    pheromone = np.ones((3, 3))  # Exemple pour un problème à 3 objectifs

    for _ in range(num_iterations):
        delta_pheromone = np.random.rand(3, 3)
        pheromone += delta_pheromone  # Mise à jour des phéromones
    best_solution = np.argmax(pheromone, axis=1)
    return best_solution

# Main logic to execute the algorithm
if algorithm_choice == "NSGA-II":
    nb_generations = st.sidebar.slider("Nombre de générations", 10, 100, 50)
    taille_population = st.sidebar.slider("Taille de la population", 10, 100, 50)
    st.subheader("Évolution des générations")
    pop, logbook = run_nsga2(taille_population, nb_generations)
    gen = logbook.select("gen")
    max_values = np.array(logbook.select("max"))
    creativity, ergonomics, efficiency = zip(*max_values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gen, y=creativity, mode='lines+markers', name="Créativité"))
    fig.add_trace(go.Scatter(x=gen, y=ergonomics, mode='lines+markers', name="Ergonomie"))
    fig.add_trace(go.Scatter(x=gen, y=efficiency, mode='lines+markers', name="Efficacité"))
    fig.update_layout(title="Évolution des générations",
                        xaxis_title="Générations",
                        yaxis_title="Valeurs",
                        template="plotly_dark")
    st.plotly_chart(fig)
    
elif algorithm_choice == "Pareto":
    taille_population = st.sidebar.slider("Taille de la population", 10, 100, 50)
    pop = run_pareto(taille_population)
    st.subheader("Pareto Front")
    front = np.array([ind.fitness.values for ind in pop])

    fig = px.scatter_3d(x=front[:, 0], y=front[:, 1], z=front[:, 2], color=front[:, 0],
                            labels={'x': 'Créativité', 'y': 'Ergonomie', 'z': 'Efficacité'},
                            title="Front de Pareto en 3D")
    st.plotly_chart(fig)
    
elif algorithm_choice == "AHP":
    ahp_value1 = st.sidebar.slider("Comparaison Créativité/Ergonomie", 0.1, 5.0, 3.0, step=0.1)
    ahp_value2 = st.sidebar.slider("Comparaison Créativité/Efficacité", 0.1, 5.0, 2.0, step=0.1)
    ahp_value3 = st.sidebar.slider("Comparaison Ergonomie/Efficacité", 0.1, 5.0, 0.5, step=0.1)

    matrix = np.array([[1, ahp_value1, ahp_value2],
                       [1/ahp_value1, 1, ahp_value3],
                       [1/ahp_value2, 1/ahp_value3, 1]])

    weights = run_ahp(matrix)
    st.write("AHP Weights:", weights)

        
    fig = px.bar(x=["Créativité", "Ergonomie", "Efficacité"],
                    y=weights, text=weights, labels={'x': 'Critères', 'y': 'Poids'},
                    title="AHP Criteria Weights")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig)
    
elif algorithm_choice == "ACO":
    num_ants = st.sidebar.slider("Nombre de fourmis", 5, 50, 10)
    num_iterations = st.sidebar.slider("Nombre d’itérations", 10, 100, 50)
    best_path = run_aco()
    st.write("ACO best path:", best_path)
        # Create a graph (example)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Example graph edges

    # Get positions using spring layout
    pos = nx.spring_layout(G)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)

    # Highlight best path
    path_edges = list(zip(best_path, best_path[1:]))  # Convert best path into edges
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

    st.pyplot(fig)
