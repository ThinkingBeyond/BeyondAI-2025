from manim import *
import networkx as nx
import numpy as np
import random



# Set seed for reproducibility of the graph layout and initial coloring
random.seed(42)
np.random.seed(42)

class OversmoothingDemo(Scene):
    def construct(self):
        # =================Configuration=================
        N_NODES = 20
        PROB_EDGE = 0.25
        N_ITERATIONS = 5 # Number of GNN "layers" to simulate
        
        # Colors
        COLOR_MAJORITY = RED
        
        
        COLOR_MINORITY = BLUE
        COLOR_MESSAGE = YELLOW
        MINORITY_RATIO = 0.2 # 20% blue nodes

        # Mixing factor: How much a node takes on neighbor colors per step.
        # Higher = faster smoothing.
        MIXING_FACTOR = 0.6 
        # ===============================================


        # 1. Generate the Graph Structure using NetworkX
        # We use Erdos-Renyi graph for random connections
        nx_graph = nx.erdos_renyi_graph(N_NODES, PROB_EDGE)
        
        # Ensure graph is connected for better visualization of total smoothing
        while not nx.is_connected(nx_graph):
             nx_graph = nx.erdos_renyi_graph(N_NODES, PROB_EDGE)

        # Compute positions using a spring layout (NetworkX returns 2D positions)
        positions_2d = nx.spring_layout(nx_graph, scale=2.5, seed=42)
        # Convert 2D positions to 3D points (z=0) for Manim
        positions = {node: np.array([pos[0], pos[1], 0.0]) for node, pos in positions_2d.items()}

        # 2. Assign Initial Colors
        node_colors_hex = {}
        node_colors_rgb = {} # Keep track of RGB numeric values for averaging

        for node in nx_graph.nodes():
            if random.random() < MINORITY_RATIO:
                c_hex = COLOR_MINORITY
            else:
                c_hex = COLOR_MAJORITY
            
            node_colors_hex[node] = c_hex
            node_colors_rgb[node] = np.array(color_to_rgb(c_hex))


        # 3. Create Manim Graph Mobject
        # We use Manim's built-in Graph mobject for easy handling of nodes and edges
        manim_graph = Graph(
            nx_graph.nodes(),
            nx_graph.edges(),
            layout=positions,
            # Use fill_color so nodes appear filled with the assigned color
            vertex_config={
                v: {"fill_color": node_colors_hex[v], "radius": 0.2, "fill_opacity": 1}
                for v in nx_graph.nodes()
            },
            edge_config={
                e: {"stroke_color": GRAY, "stroke_width": 2} 
                for e in nx_graph.edges()
            }
        )

        # Initial Title text
        title = Text("Initial State: Distinct Features (Heterophily)", font_size=24)
        title.to_edge(UP)

        # --- Animation Start ---
        self.play(Write(title))
        self.play(Create(manim_graph), run_time=2)
        self.wait(1)

        # 4. The Main Loop: Simulating Message Passing and Smoothing
        for i in range(1, N_ITERATIONS + 1):
            
            # Update title indicating the layer depth
            new_title = Text(f"GNN Layer {i}: Message Passing & Aggregation", font_size=24).to_edge(UP)
            self.play(Transform(title, new_title))

            # --- Step A: Visualize Message Passing ---
            messages_group = VGroup()
            message_animations = []

            for edge in nx_graph.edges():
                u, v = edge
                # Get actual manim objects for start and end points
                start_node = manim_graph.vertices[u]
                end_node = manim_graph.vertices[v]
                
                # Create a small dot representing a message/feature vector
                msg_dot = Dot(point=start_node.get_center(), radius=0.08, color=COLOR_MESSAGE)
                messages_group.add(msg_dot)
                
                # Animate moving from u to v
                message_animations.append(MoveAlongPath(msg_dot, Line(start_node.get_center(), end_node.get_center())))
                
                # Optional: bi-directional messages for undirected graphs
                msg_dot_back = Dot(point=end_node.get_center(), radius=0.08, color=COLOR_MESSAGE)
                messages_group.add(msg_dot_back)
                message_animations.append(MoveAlongPath(msg_dot_back, Line(end_node.get_center(), start_node.get_center())))

            # Add the message dots to the scene, then play message passing animation quickly
            self.add(messages_group)
            self.play(*message_animations, run_time=1.5, rate_func=linear)
            self.remove(messages_group) # Clean up message dots


            # --- Step B: Calculate and Animate Color Smoothing (Aggregation) ---
            
            new_node_colors_rgb = {}
            color_change_animations = []

            for node in nx_graph.nodes():
                # Get neighbors
                neighbors = list(nx_graph.neighbors(node))
                
                # Calculate average RGB of neighbors
                if neighbors:
                    neighbor_rgbs = np.array([node_colors_rgb[n] for n in neighbors])
                    avg_neighbor_rgb = np.mean(neighbor_rgbs, axis=0)
                else:
                    # Should not happen due to connectivity check, but good practice
                    avg_neighbor_rgb = node_colors_rgb[node]

                # Update rule: blend current color with average neighbor color
                # new_color = (1 - mixing) * old + mixing * neighbor_avg
                current_rgb = node_colors_rgb[node]
                new_rgb = (1 - MIXING_FACTOR) * current_rgb + MIXING_FACTOR * avg_neighbor_rgb
                
                # Store for next iteration
                new_node_colors_rgb[node] = new_rgb
                
                # Convert back to hex for Manim animation (ensure values in 0..1)
                # and use the helper to produce a hex string compatible with set_color
                new_hex = custom_rgb_to_hex(np.clip(new_rgb, 0.0, 1.0))

                # Create animation to change fill color
                color_change_animations.append(
                    manim_graph.vertices[node].animate.set_fill(new_hex)
                )

            # Update the RGB state dictionary for the next loop iteration
            node_colors_rgb = new_node_colors_rgb
            
            # Play color convergence animation
            self.play(*color_change_animations, run_time=1.5)
            self.wait(0.5)


        # 5. Final State Text
        final_title = Text("Final State: Oversmoothed (Indistinguishable Features)", 
                           font_size=24, color=RED_B).to_edge(UP)
        self.play(Transform(title, final_title))
        self.wait(3)

# Helper function just in case manim version doesn't have rgb_to_color handy or it behaves differently. 
# Manim's internal color_to_rgb returns numpy array [r,g,b] between 0 and 1.
# Manim's rgb_to_color expects the same.
# The code uses Manim's built-ins, but this logic is what is happening under the hood.
def custom_rgb_to_hex(rgb_array):
    """Converts a numpy RGB array [0.0-1.0] to hex string."""
    r, g, b = rgb_array
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))