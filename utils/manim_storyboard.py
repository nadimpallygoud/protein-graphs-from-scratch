"""
Teaching-oriented Manim scenes for the GNN protein explainability pipeline.

The scenes are designed as short visual explanations:
question -> mechanism -> takeaway.
"""

from manim import *
import numpy as np


config.background_color = "#0B1020"

NODE_COLOR = BLUE_C
NODE_ALT_COLOR = TEAL_C
NODE_MUTED = BLUE_E
EDGE_COLOR = GREY_B
EDGE_SOFT = GREY_D
ACCENT_COLOR = YELLOW_C
RELEVANCE_COLOR = RED_C
SECONDARY_RELEVANCE = ORANGE
SUCCESS_COLOR = GREEN_C
PANEL_FILL = "#141B30"
MUTED_TEXT = GREY_A


class TeachingScene(Scene):
    """Shared layout helpers for the teaching GIFs."""

    def make_header(self, stage, title, subtitle):
        kicker = Text(stage, font_size=20, color=ACCENT_COLOR)
        main = Text(title, font_size=38, weight=BOLD)
        sub = Text(subtitle, font_size=20, color=MUTED_TEXT)
        text = VGroup(kicker, main, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rule = Line(LEFT * 3.9, RIGHT * 3.9, color=NODE_ALT_COLOR, stroke_width=3)
        header = VGroup(text, rule).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        header.to_edge(UP, buff=0.3)
        return header

    def make_panel(self, title, lines, accent=ACCENT_COLOR):
        heading = Text(title, font_size=22, color=accent, weight=BOLD)
        body = VGroup(*[Text(line, font_size=18) for line in lines]).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.12,
        )
        content = VGroup(heading, body).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        frame = SurroundingRectangle(
            content,
            buff=0.22,
            corner_radius=0.18,
            color=GREY_B,
            stroke_width=1.5,
        )
        frame.set_fill(PANEL_FILL, opacity=0.92)
        return VGroup(frame, content)

    def make_takeaway(self, lines):
        return self.make_panel("Takeaway", lines, accent=SUCCESS_COLOR)

    def make_labeled_node(
        self,
        label,
        position,
        color=NODE_COLOR,
        radius=0.2,
        label_direction=DOWN,
        label_buff=0.12,
        label_size=20,
    ):
        dot = Dot(position, radius=radius, color=color)
        text = Text(label, font_size=label_size).next_to(dot, label_direction, buff=label_buff)
        return VGroup(dot, text)

    def make_feature_strip(self, values, fill_colors, scale=0.52, font_size=18):
        cells = VGroup()
        for value, fill in zip(values, fill_colors):
            box = Square(side_length=scale, color=WHITE, stroke_width=1.4)
            box.set_fill(fill, opacity=0.82)
            label = Text(str(value), font_size=font_size).move_to(box.get_center())
            cells.add(VGroup(box, label))
        cells.arrange(RIGHT, buff=0.05)
        return cells

    def make_value_grid(self, rows, cell_width=0.58, cell_height=0.5, font_size=20):
        row_groups = VGroup()
        for row in rows:
            cells = VGroup()
            for value in row:
                box = RoundedRectangle(
                    corner_radius=0.08,
                    width=cell_width,
                    height=cell_height,
                    color=GREY_B,
                    stroke_width=1.2,
                ).set_fill("#10172A", opacity=0.96)
                label = Text(str(value), font_size=font_size).move_to(box.get_center())
                cells.add(VGroup(box, label))
            cells.arrange(RIGHT, buff=0.08)
            row_groups.add(cells)
        row_groups.arrange(DOWN, buff=0.08)
        return row_groups

    def make_formula(self, text, font_size=24, color=WHITE):
        return Text(text, font_size=font_size, color=color, font="Consolas")

    def place_panel(self, panel, corner, shift_vector=ORIGIN):
        panel.to_corner(corner, buff=0.4)
        if shift_vector is not ORIGIN:
            panel.shift(shift_vector)
        return panel

    def fade_all(self, run_time=0.6):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)


class GraphBasicsScene(TeachingScene):
    """Stage 1: adjacency as the bookkeeping object for a graph."""

    def construct(self):
        header = self.make_header(
            "Stage 1",
            "Graph basics",
            "A graph is entities plus the relationships between them.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "How do we turn objects and",
                    "relationships into something",
                    "a model can compute with?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        labels = ["A", "B", "C", "D"]
        positions = {
            "A": np.array([-5.0, 0.9, 0.0]),
            "B": np.array([-3.5, 2.2, 0.0]),
            "C": np.array([-3.3, -0.6, 0.0]),
            "D": np.array([-1.7, -1.9, 0.0]),
        }
        nodes = VGroup(
            *[
                self.make_labeled_node(label, positions[label], label_direction=UP)
                for label in labels
            ]
        )
        self.play(LaggedStart(*[FadeIn(node, shift=UP * 0.2) for node in nodes], lag_ratio=0.12))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Each node is one object we care",
                    "about.",
                    "Later, a node can be a residue,",
                    "an atom, or a learned state.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        edge_pairs = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]
        edge_lookup = {}
        edges = VGroup()
        for left, right in edge_pairs:
            line = Line(positions[left], positions[right], color=EDGE_COLOR, stroke_width=3)
            edges.add(line)
            edge_lookup[(left, right)] = line
            edge_lookup[(right, left)] = line

        self.play(LaggedStart(*[Create(edge) for edge in edges], lag_ratio=0.15))
        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "Edges say who can interact or",
                    "exchange information.",
                    "Without edges, the model has no",
                    "structural context to use.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        matrix = self.make_value_grid(
            [["0", "1", "1", "0"], ["1", "0", "1", "0"], ["1", "1", "0", "1"], ["0", "0", "1", "0"]],
            font_size=20,
        )
        matrix_block = VGroup(
            Text("Adjacency matrix A", font_size=22, color=NODE_ALT_COLOR),
            matrix,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        matrix_block.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.45)
        self.play(FadeIn(matrix_block, shift=LEFT))

        row_box = SurroundingRectangle(matrix[2], color=ACCENT_COLOR, buff=0.08)
        self.play(Create(row_box))
        self.play(
            edge_lookup[("C", "A")].animate.set_color(ACCENT_COLOR).set_stroke(width=5),
            edge_lookup[("C", "B")].animate.set_color(ACCENT_COLOR).set_stroke(width=5),
            edge_lookup[("C", "D")].animate.set_color(ACCENT_COLOR).set_stroke(width=5),
            nodes[2][0].animate.set_color(ACCENT_COLOR).scale(1.15),
        )

        step_three = self.place_panel(
            self.make_panel(
                "Step 3",
                [
                    "Row C lists every node that C can",
                    "directly see.",
                    "That matrix is the bookkeeping",
                    "used by later GNN layers.",
                ],
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_three))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "A graph is not just a picture.",
                    "It is a relation map that tells a",
                    "GNN where messages are allowed",
                    "to travel.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway), FadeOut(row_box))
        self.wait(0.9)
        self.fade_all()


class GCNMessagePassingScene(TeachingScene):
    """Stage 2: intuitive message passing with normalization."""

    def construct(self):
        header = self.make_header(
            "Stage 2",
            "Message passing",
            "A node updates by mixing its own features with normalized neighbor messages.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "What does a GNN layer actually",
                    "do to one node state?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        target = self.make_labeled_node("target", np.array([-2.2, -0.6, 0.0]), color=NODE_COLOR, radius=0.28)
        neighbors = VGroup(
            self.make_labeled_node("n1", np.array([-4.7, 1.8, 0.0]), color=NODE_MUTED, label_direction=UP),
            self.make_labeled_node("n2", np.array([-2.1, 2.5, 0.0]), color=NODE_MUTED, label_direction=UP),
            self.make_labeled_node("n3", np.array([0.4, 1.5, 0.0]), color=NODE_MUTED, label_direction=UP),
        )
        edges = VGroup(
            *[
                Line(neighbor[0].get_center(), target[0].get_center(), color=EDGE_COLOR, stroke_width=3)
                for neighbor in neighbors
            ]
        )
        self.play(FadeIn(target), FadeIn(neighbors), Create(edges))

        neighbor_vectors = VGroup(
            self.make_feature_strip(["0.9", "0.1", "0.2"], [BLUE_E, TEAL_E, GREEN_E]),
            self.make_feature_strip(["0.2", "0.8", "0.4"], [BLUE_E, TEAL_E, GREEN_E]),
            self.make_feature_strip(["0.6", "0.3", "0.9"], [BLUE_E, TEAL_E, GREEN_E]),
        )
        own_vector = self.make_feature_strip(["0.5", "0.2", "0.7"], [BLUE_E, TEAL_E, GREEN_E])
        for index, strip in enumerate(neighbor_vectors):
            strip.next_to(neighbors[index][0], UP, buff=0.18)
        own_vector.next_to(target[0], DOWN, buff=0.22)
        self.play(FadeIn(neighbor_vectors), FadeIn(own_vector))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Every node starts with its own",
                    "feature vector.",
                    "The vector can store chemistry,",
                    "geometry, or learned signals.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        self_loop = Circle(radius=0.55, color=ACCENT_COLOR, stroke_width=3).move_to(target[0].get_center())
        self.play(Create(self_loop))

        weights = VGroup()
        for edge in edges:
            weights.add(Text("1/4", font_size=18, color=ACCENT_COLOR).move_to(edge.get_center() + RIGHT * 0.2))
        weights.add(Text("1/4", font_size=18, color=ACCENT_COLOR).move_to(self_loop.get_center() + DOWN * 0.65))
        self.play(FadeIn(weights))

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "Messages arrive from neighbors",
                    "and from the node itself.",
                    "The self-loop keeps the original",
                    "signal in the update.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        aggregator = RoundedRectangle(
            corner_radius=0.18,
            width=2.8,
            height=1.2,
            color=GREY_B,
            stroke_width=1.6,
        ).set_fill(PANEL_FILL, opacity=0.9)
        aggregator.move_to(np.array([2.7, -0.1, 0.0]))
        aggregator_label = Text("weighted sum", font_size=22, color=ACCENT_COLOR).move_to(aggregator)
        self.play(FadeIn(aggregator), FadeIn(aggregator_label))

        messages = VGroup(*[strip.copy() for strip in neighbor_vectors], own_vector.copy())
        targets = [
            aggregator.get_center() + UP * 0.33 + LEFT * 0.65,
            aggregator.get_center() + UP * 0.33 + RIGHT * 0.65,
            aggregator.get_center() + DOWN * 0.33 + LEFT * 0.65,
            aggregator.get_center() + DOWN * 0.33 + RIGHT * 0.65,
        ]
        self.play(
            *[messages[index].animate.move_to(targets[index]) for index in range(len(messages))],
            run_time=1.8,
        )

        equation = self.make_formula(
            "h_new = (h_self + h_n1 + h_n2 + h_n3) / 4",
            font_size=22,
            color=ACCENT_COLOR,
        ).to_edge(DOWN, buff=0.55)
        self.play(Write(equation))

        final_vector = self.make_feature_strip(["0.55", "0.35", "0.55"], [BLUE_C, TEAL_C, GREEN_C])
        final_vector.next_to(target[0], RIGHT, buff=1.5)
        final_label = Text("updated state", font_size=20, color=SUCCESS_COLOR).next_to(final_vector, DOWN, buff=0.15)
        self.play(Transform(messages, final_vector), FadeIn(final_label))
        self.play(target[0].animate.set_color(NODE_ALT_COLOR).scale(1.12))

        step_three = self.place_panel(
            self.make_panel(
                "Step 3",
                [
                    "Normalization shares credit",
                    "between all incoming sources.",
                    "That prevents high-degree nodes",
                    "from overwhelming the update.",
                ],
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_three))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "A GCN layer is a controlled mixing",
                    "operation: own signal plus",
                    "neighbor context, scaled so the",
                    "update stays stable.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class FeatureVectorUpdateScene(TeachingScene):
    """Deep-dive view of a feature update with explicit dimensions."""

    def construct(self):
        header = self.make_header(
            "Deep Dive",
            "Feature vector update",
            "The embedding update is an actual numerical combination, not a vague color shift.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "What is being aggregated during",
                    "message passing?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        legend = self.make_panel(
            "Vector meaning",
            [
                "slot 1: chemistry",
                "slot 2: charge",
                "slot 3: geometry",
            ],
            accent=NODE_ALT_COLOR,
        )
        legend.scale(0.78)
        legend.to_corner(UL, buff=0.45).shift(DOWN * 0.95)
        self.play(FadeIn(legend, shift=RIGHT))

        target = self.make_labeled_node("Residue v", np.array([0.0, -1.4, 0.0]), color=NODE_COLOR, radius=0.26)
        neighbors = VGroup(
            self.make_labeled_node("u1", np.array([-4.2, 1.6, 0.0]), color=NODE_MUTED, label_direction=UP),
            self.make_labeled_node("u2", np.array([0.0, 2.5, 0.0]), color=NODE_MUTED, label_direction=UP),
            self.make_labeled_node("u3", np.array([4.2, 1.6, 0.0]), color=NODE_MUTED, label_direction=UP),
        )
        edges = VGroup(*[Line(node[0].get_center(), target[0].get_center(), color=EDGE_COLOR, stroke_width=3) for node in neighbors])
        self.play(FadeIn(target), FadeIn(neighbors), Create(edges))

        own_vector = self.make_feature_strip(["0.6", "0.1", "0.7"], [BLUE_E, TEAL_E, GREEN_E])
        own_vector.next_to(target[0], DOWN, buff=0.25)
        neighbor_vectors = VGroup(
            self.make_feature_strip(["0.9", "0.3", "0.2"], [BLUE_E, TEAL_E, GREEN_E]),
            self.make_feature_strip(["0.2", "0.8", "0.5"], [BLUE_E, TEAL_E, GREEN_E]),
            self.make_feature_strip(["0.4", "0.2", "0.9"], [BLUE_E, TEAL_E, GREEN_E]),
        )
        for index, vector in enumerate(neighbor_vectors):
            vector.next_to(neighbors[index][0], UP, buff=0.18)
        self.play(FadeIn(own_vector), FadeIn(neighbor_vectors))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Each slot carries one dimension",
                    "of the node state.",
                    "The model updates all dimensions",
                    "together, not one at a time.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        aggregator = RoundedRectangle(
            corner_radius=0.18,
            width=3.1,
            height=1.3,
            color=GREY_B,
            stroke_width=1.6,
        ).set_fill(PANEL_FILL, opacity=0.9)
        aggregator.move_to(np.array([0.0, 0.25, 0.0]))
        aggregator_text = Text("mean(self + neighbors)", font_size=22, color=ACCENT_COLOR).move_to(aggregator)
        self.play(FadeIn(aggregator), FadeIn(aggregator_text))

        traveling = VGroup(*[vector.copy() for vector in neighbor_vectors], own_vector.copy())
        targets = [
            aggregator.get_center() + UP * 0.32 + LEFT * 0.8,
            aggregator.get_center() + UP * 0.32 + RIGHT * 0.8,
            aggregator.get_center() + DOWN * 0.32 + LEFT * 0.8,
            aggregator.get_center() + DOWN * 0.32 + RIGHT * 0.8,
        ]
        self.play(
            *[traveling[index].animate.move_to(targets[index]) for index in range(len(traveling))],
            run_time=1.8,
        )

        equation = self.make_formula(
            "h_new = mean(h_self, h_u1, h_u2, h_u3)",
            font_size=22,
            color=ACCENT_COLOR,
        ).to_edge(DOWN, buff=0.55)
        self.play(Write(equation))

        result = self.make_feature_strip(["0.53", "0.35", "0.58"], [BLUE_C, TEAL_C, GREEN_C])
        result.next_to(target[0], RIGHT, buff=2.0)
        result_label = Text("new embedding", font_size=20, color=SUCCESS_COLOR).next_to(result, DOWN, buff=0.15)
        self.play(Transform(traveling, result), FadeIn(result_label))
        self.play(target[0].animate.set_color(NODE_ALT_COLOR).scale(1.12))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "The learned embedding is a compact",
                    "summary of local chemistry and",
                    "structure after neighbor mixing.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class PyGEdgeIndexScene(TeachingScene):
    """Stage 3: sparse edge storage in PyTorch Geometric."""

    def construct(self):
        header = self.make_header(
            "Stage 3",
            "PyG edge_index",
            "Sparse storage keeps only the edges that actually exist.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "How does PyTorch Geometric store",
                    "graph connectivity in code?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        positions = {
            0: np.array([-4.8, 0.8, 0.0]),
            1: np.array([-3.0, 2.0, 0.0]),
            2: np.array([-2.5, -1.2, 0.0]),
        }
        nodes = VGroup(*[self.make_labeled_node(str(index), position, label_direction=DOWN) for index, position in positions.items()])
        self.play(FadeIn(nodes))

        matrix = self.make_value_grid([["0", "1", "1", "2"], ["1", "0", "2", "1"]], font_size=20)
        title = Text("edge_index", font_size=24, color=NODE_ALT_COLOR)
        matrix_block = VGroup(title, matrix).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        matrix_block.move_to(np.array([2.8, 0.1, 0.0]))
        row_labels = VGroup(
            Text("source", font_size=18, color=MUTED_TEXT).next_to(matrix[0], LEFT, buff=0.18),
            Text("target", font_size=18, color=MUTED_TEXT).next_to(matrix[1], LEFT, buff=0.18),
        )
        self.play(FadeIn(matrix_block, shift=LEFT), FadeIn(row_labels))

        arrows = VGroup(
            Arrow(positions[0], positions[1], buff=0.3, color=EDGE_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.13),
            Arrow(positions[1], positions[0], buff=0.3, color=EDGE_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.13),
            Arrow(positions[1], positions[2], buff=0.3, color=EDGE_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.13),
            Arrow(positions[2], positions[1], buff=0.3, color=EDGE_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.13),
        )

        columns = [
            VGroup(matrix[0][0], matrix[1][0]),
            VGroup(matrix[0][1], matrix[1][1]),
            VGroup(matrix[0][2], matrix[1][2]),
            VGroup(matrix[0][3], matrix[1][3]),
        ]

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Each column is one directed edge:",
                    "source index on top,",
                    "target index below.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        for arrow, column in zip(arrows, columns):
            box = SurroundingRectangle(column, color=ACCENT_COLOR, buff=0.08)
            self.play(Create(arrow), Create(box), run_time=0.7)
            self.play(FadeOut(box), run_time=0.2)

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "An undirected edge is usually",
                    "stored twice.",
                    "That lets message passing work",
                    "in both directions.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))
        pair_box = SurroundingRectangle(VGroup(columns[0], columns[1]), color=SUCCESS_COLOR, buff=0.1)
        self.play(Create(pair_box))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "PyG avoids a dense n by n matrix.",
                    "It stores only real connections,",
                    "which is why it scales to larger",
                    "graphs much more efficiently.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway), FadeOut(pair_box))
        self.wait(0.9)
        self.fade_all()


class ProteinBackboneScene(TeachingScene):
    """Visual bridge from sequence order to folded 3D contacts."""

    def construct(self):
        header = self.make_header(
            "Stage 4",
            "Sequence vs folded structure",
            "Protein graphs must follow 3D contacts, not only sequence order.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "Why is a protein graph not just",
                    "the amino-acid chain?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        sequence_positions = [
            np.array([-5.5, 1.7, 0.0]),
            np.array([-4.2, 1.7, 0.0]),
            np.array([-2.9, 1.7, 0.0]),
            np.array([-1.6, 1.7, 0.0]),
            np.array([-0.3, 1.7, 0.0]),
            np.array([1.0, 1.7, 0.0]),
        ]
        folded_positions = [
            np.array([-4.8, -1.2, 0.0]),
            np.array([-3.5, 0.1, 0.0]),
            np.array([-2.1, -0.6, 0.0]),
            np.array([-0.8, -1.8, 0.0]),
            np.array([0.5, -0.2, 0.0]),
            np.array([1.3, -1.4, 0.0]),
        ]

        sequence_chain = VGroup(*[Line(sequence_positions[index], sequence_positions[index + 1], color=EDGE_COLOR, stroke_width=3) for index in range(5)])
        sequence_nodes = VGroup(*[self.make_labeled_node(str(index + 1), position, label_direction=UP) for index, position in enumerate(sequence_positions)])
        sequence_label = Text("sequence neighbors", font_size=22, color=MUTED_TEXT).next_to(sequence_chain, UP, buff=0.25)
        self.play(Create(sequence_chain), FadeIn(sequence_nodes), Write(sequence_label))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Along the chain, residue 2 sits",
                    "next to 3, and 5 sits next to 6.",
                    "Sequence order is only one view",
                    "of the protein.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        fold_line = VMobject(color=EDGE_COLOR, stroke_width=4)
        fold_line.set_points_as_corners(folded_positions)
        folded_nodes = VGroup(*[self.make_labeled_node(str(index + 1), position, label_direction=DOWN) for index, position in enumerate(folded_positions)])
        folded_label = Text("same residues after folding", font_size=22, color=MUTED_TEXT).next_to(fold_line, DOWN, buff=0.3)
        self.play(TransformFromCopy(sequence_chain, fold_line), TransformFromCopy(sequence_nodes, folded_nodes), FadeIn(folded_label))

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "After folding, sequence-distant",
                    "residues can become neighbors in",
                    "physical space.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        focus_circle = Circle(radius=1.1, color=ACCENT_COLOR, stroke_width=3).move_to(folded_nodes[1][0].get_center())
        contact_edge = Line(
            folded_nodes[1][0].get_center(),
            folded_nodes[4][0].get_center(),
            color=RELEVANCE_COLOR,
            stroke_width=5,
        )
        sequence_gap = DashedLine(sequence_nodes[1][0].get_center(), sequence_nodes[4][0].get_center(), dash_length=0.12, color=SECONDARY_RELEVANCE)
        gap_label = Text("far in sequence", font_size=18, color=SECONDARY_RELEVANCE).next_to(sequence_gap, DOWN, buff=0.12)
        contact_label = Text("close in 3D", font_size=18, color=RELEVANCE_COLOR).next_to(contact_edge, RIGHT, buff=0.18)
        self.play(Create(sequence_gap), FadeIn(gap_label))
        self.play(Create(focus_circle), Create(contact_edge), FadeIn(contact_label))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "Residue graphs exist to preserve",
                    "those folded contacts.",
                    "That is how a protein GNN sees",
                    "long-range structural effects.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class ProteinDistanceGraphScene(TeachingScene):
    """Stage 4 detail: contact threshold becomes graph edges."""

    def construct(self):
        header = self.make_header(
            "Stage 4",
            "Residue graph construction",
            "A distance threshold decides which residues become graph neighbors.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "How do we convert 3D coordinates",
                    "into edges for the graph?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        positions = [
            np.array([-4.5, 0.1, 0.0]),
            np.array([-3.2, 1.4, 0.0]),
            np.array([-1.8, 0.2, 0.0]),
            np.array([-0.5, -1.3, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, -1.2, 0.0]),
        ]
        backbone = VMobject(color=EDGE_SOFT, stroke_width=4)
        backbone.set_points_as_corners(positions)
        residues = VGroup(*[self.make_labeled_node(f"R{index + 1}", point, label_direction=DOWN, label_size=18) for index, point in enumerate(positions)])
        self.play(Create(backbone), FadeIn(residues))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Each residue becomes one node",
                    "with a 3D coordinate and feature",
                    "vector.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        center = residues[2][0]
        threshold = Circle(radius=1.75, color=ACCENT_COLOR, stroke_width=3).move_to(center.get_center())
        self.play(Create(threshold))

        nearby_edges = VGroup(
            Line(residues[1][0].get_center(), center.get_center(), color=EDGE_COLOR, stroke_width=4),
            Line(residues[3][0].get_center(), center.get_center(), color=EDGE_COLOR, stroke_width=4),
            Line(residues[4][0].get_center(), center.get_center(), color=EDGE_COLOR, stroke_width=4),
        )
        far_marker = Cross(residues[5][0], color=SECONDARY_RELEVANCE, stroke_width=5).scale(0.4)
        self.play(LaggedStart(*[Create(edge) for edge in nearby_edges], lag_ratio=0.15), FadeIn(far_marker))

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "Only residues inside the distance",
                    "threshold receive contact edges.",
                    "Distant residues stay disconnected.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        graph_label = Text("contact graph around R3", font_size=20, color=SUCCESS_COLOR).next_to(threshold, UP, buff=0.15)
        self.play(FadeIn(graph_label))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "The graph is a learned simplification",
                    "of the structure: enough geometry",
                    "to model residue interactions, but",
                    "compact enough for GNN layers.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class GraphPoolingScene(TeachingScene):
    """Stage 5: graph-level pooling for protein classification."""

    def construct(self):
        header = self.make_header(
            "Stage 5",
            "Graph-level prediction",
            "Pooling converts many node embeddings into one protein-level representation.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "How does a node-wise model make",
                    "one prediction for the whole",
                    "protein graph?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        graph_positions = [
            np.array([-4.8, 0.6, 0.0]),
            np.array([-3.8, 2.0, 0.0]),
            np.array([-2.4, 1.2, 0.0]),
            np.array([-2.2, -0.6, 0.0]),
            np.array([-3.7, -1.5, 0.0]),
        ]
        nodes = VGroup(*[Dot(point, radius=0.18, color=NODE_COLOR) for point in graph_positions])
        edges = VGroup(
            Line(graph_positions[0], graph_positions[1], color=EDGE_COLOR, stroke_width=3),
            Line(graph_positions[1], graph_positions[2], color=EDGE_COLOR, stroke_width=3),
            Line(graph_positions[2], graph_positions[3], color=EDGE_COLOR, stroke_width=3),
            Line(graph_positions[3], graph_positions[4], color=EDGE_COLOR, stroke_width=3),
            Line(graph_positions[4], graph_positions[0], color=EDGE_COLOR, stroke_width=3),
        )
        node_states = VGroup(
            self.make_feature_strip(["0.8", "0.2", "0.5"], [BLUE_E, TEAL_E, GREEN_E], scale=0.42, font_size=15),
            self.make_feature_strip(["0.3", "0.7", "0.6"], [BLUE_E, TEAL_E, GREEN_E], scale=0.42, font_size=15),
            self.make_feature_strip(["0.6", "0.4", "0.3"], [BLUE_E, TEAL_E, GREEN_E], scale=0.42, font_size=15),
            self.make_feature_strip(["0.5", "0.5", "0.8"], [BLUE_E, TEAL_E, GREEN_E], scale=0.42, font_size=15),
            self.make_feature_strip(["0.7", "0.1", "0.4"], [BLUE_E, TEAL_E, GREEN_E], scale=0.42, font_size=15),
        )
        for index, state in enumerate(node_states):
            state.next_to(nodes[index], RIGHT, buff=0.14)

        graph_label = Text("node embeddings", font_size=22, color=MUTED_TEXT).next_to(nodes, UP, buff=0.35)
        self.play(Create(edges), FadeIn(nodes), FadeIn(node_states), Write(graph_label))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "After several GNN layers, each",
                    "node holds a local structural",
                    "summary of its neighborhood.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        pooling_box = RoundedRectangle(
            corner_radius=0.18,
            width=2.5,
            height=1.2,
            color=GREY_B,
            stroke_width=1.6,
        ).set_fill(PANEL_FILL, opacity=0.9)
        pooling_box.move_to(np.array([1.0, 0.1, 0.0]))
        pooling_label = Text("global mean", font_size=24, color=ACCENT_COLOR).move_to(pooling_box)
        self.play(FadeIn(pooling_box), FadeIn(pooling_label))

        arrows = VGroup(*[Arrow(state.get_right(), pooling_box.get_left(), buff=0.15, color=EDGE_COLOR, stroke_width=3, max_tip_length_to_length_ratio=0.12) for state in node_states])
        self.play(LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.1))

        pooled = self.make_feature_strip(["0.58", "0.38", "0.52"], [BLUE_C, TEAL_C, GREEN_C], scale=0.55)
        pooled.next_to(pooling_box, RIGHT, buff=1.0)
        pooled_label = Text("graph embedding", font_size=20, color=SUCCESS_COLOR).next_to(pooled, DOWN, buff=0.15)
        self.play(FadeIn(pooled), FadeIn(pooled_label))

        classifier = RoundedRectangle(corner_radius=0.16, width=1.7, height=0.9, color=GREY_B).set_fill(PANEL_FILL, opacity=0.92)
        classifier.next_to(pooled, RIGHT, buff=1.0)
        classifier_label = Text("MLP", font_size=22).move_to(classifier)
        output = self.make_panel("Prediction", ["lysozyme", "score: 0.92"], accent=RELEVANCE_COLOR)
        output.scale(0.78)
        output.next_to(classifier, RIGHT, buff=0.7)
        connector = Arrow(pooled.get_right(), classifier.get_left(), buff=0.12, color=EDGE_COLOR, stroke_width=3, max_tip_length_to_length_ratio=0.12)
        connector_two = Arrow(classifier.get_right(), output.get_left(), buff=0.12, color=EDGE_COLOR, stroke_width=3, max_tip_length_to_length_ratio=0.12)
        self.play(FadeIn(classifier), FadeIn(classifier_label), Create(connector), Create(connector_two), FadeIn(output))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "Pooling is what lets a variable-size",
                    "residue graph produce one fixed-size",
                    "representation for classification.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class SaliencyVsSubgraphScene(TeachingScene):
    """Stage 6: contrast pointwise saliency with connected structural evidence."""

    def construct(self):
        header = self.make_header(
            "Stage 6",
            "Saliency vs structural evidence",
            "Some explanations score individual residues; others reveal cooperating subgraphs.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "Why is a single-node heatmap often",
                    "not enough for proteins?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        base_positions = [
            np.array([-5.2, 0.2, 0.0]),
            np.array([-4.2, 1.6, 0.0]),
            np.array([-2.8, 0.7, 0.0]),
            np.array([-2.4, -0.8, 0.0]),
            np.array([-3.8, -1.8, 0.0]),
        ]
        right_shift = RIGHT * 6.0

        left_nodes = VGroup(*[Dot(point, radius=0.18, color=NODE_COLOR) for point in base_positions])
        left_edges = VGroup(
            Line(base_positions[0], base_positions[1], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[1], base_positions[2], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[2], base_positions[3], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[3], base_positions[4], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[4], base_positions[0], color=EDGE_COLOR, stroke_width=3),
        )
        right_nodes = left_nodes.copy().shift(right_shift)
        right_edges = left_edges.copy().shift(right_shift)

        left_title = Text("saliency", font_size=24, color=SECONDARY_RELEVANCE).next_to(left_nodes, UP, buff=0.35)
        right_title = Text("connected subgraph", font_size=24, color=RELEVANCE_COLOR).next_to(right_nodes, UP, buff=0.35)
        self.play(FadeIn(left_nodes, left_edges, right_nodes, right_edges), Write(left_title), Write(right_title))

        step_one = self.place_panel(
            self.make_panel(
                "Saliency",
                [
                    "A sensitivity map can highlight",
                    "isolated nodes or features.",
                    "Useful, but it may miss the fact",
                    "that residues act together.",
                ],
                accent=SECONDARY_RELEVANCE,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        saliency_glow = VGroup(
            left_nodes[1].copy().set_color(SECONDARY_RELEVANCE).scale(1.45),
            left_nodes[3].copy().set_color(SECONDARY_RELEVANCE).scale(1.45),
        )
        self.play(FadeIn(saliency_glow))

        step_two = self.place_panel(
            self.make_panel(
                "Subgraph view",
                [
                    "A structural explanation asks",
                    "which connected interaction pattern",
                    "supports the prediction.",
                ],
                accent=RELEVANCE_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        subgraph = VGroup(
            right_nodes[0].copy().set_color(RELEVANCE_COLOR).scale(1.35),
            right_nodes[1].copy().set_color(RELEVANCE_COLOR).scale(1.35),
            right_nodes[2].copy().set_color(RELEVANCE_COLOR).scale(1.35),
            Line(right_nodes[0].get_center(), right_nodes[1].get_center(), color=RELEVANCE_COLOR, stroke_width=6),
            Line(right_nodes[1].get_center(), right_nodes[2].get_center(), color=RELEVANCE_COLOR, stroke_width=6),
        )
        self.play(FadeIn(subgraph))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "In proteins, mechanism often lives",
                    "in interacting residue sets.",
                    "That is why structure-aware",
                    "explanations are important.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class LRPNumericConservationScene(TeachingScene):
    """Stage 7 detail: show conservation numerically."""

    def construct(self):
        header = self.make_header(
            "Stage 7",
            "Relevance conservation",
            "GNN-LRP redistributes evidence while keeping the total score intact.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "What makes relevance propagation",
                    "different from raw gradients?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        output_node = Dot(np.array([4.7, 0.0, 0.0]), radius=0.36, color=RELEVANCE_COLOR)
        output_value = Text("0.92", font_size=22).move_to(output_node)
        hidden = VGroup(
            Dot(np.array([1.4, 1.8, 0.0]), radius=0.28, color=NODE_MUTED),
            Dot(np.array([1.4, -1.8, 0.0]), radius=0.28, color=NODE_MUTED),
        )
        inputs = VGroup(
            Dot(np.array([-2.0, 2.4, 0.0]), radius=0.22, color=NODE_COLOR),
            Dot(np.array([-2.0, 0.8, 0.0]), radius=0.22, color=NODE_COLOR),
            Dot(np.array([-2.0, -0.8, 0.0]), radius=0.22, color=NODE_COLOR),
            Dot(np.array([-2.0, -2.4, 0.0]), radius=0.22, color=NODE_COLOR),
        )
        self.play(FadeIn(output_node, output_value, hidden, inputs))

        edges = VGroup(
            Line(hidden[0].get_center(), output_node.get_center(), color=EDGE_SOFT, stroke_width=3),
            Line(hidden[1].get_center(), output_node.get_center(), color=EDGE_SOFT, stroke_width=3),
            Line(inputs[0].get_center(), hidden[0].get_center(), color=EDGE_SOFT, stroke_width=3),
            Line(inputs[1].get_center(), hidden[0].get_center(), color=EDGE_SOFT, stroke_width=3),
            Line(inputs[2].get_center(), hidden[1].get_center(), color=EDGE_SOFT, stroke_width=3),
            Line(inputs[3].get_center(), hidden[1].get_center(), color=EDGE_SOFT, stroke_width=3),
        )
        self.play(Create(edges))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Start from the prediction score.",
                    "LRP treats that score as evidence",
                    "that must be reassigned backward.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        split_one = VGroup(
            Text("0.55", font_size=20, color=RELEVANCE_COLOR).move_to(edges[0].get_center() + UP * 0.2),
            Text("0.37", font_size=20, color=RELEVANCE_COLOR).move_to(edges[1].get_center() + DOWN * 0.2),
        )
        eq_one = self.make_formula("0.92 = 0.55 + 0.37", font_size=24, color=ACCENT_COLOR).to_edge(DOWN, buff=0.55)
        self.play(
            edges[0].animate.set_color(RELEVANCE_COLOR).set_stroke(width=5),
            edges[1].animate.set_color(RELEVANCE_COLOR).set_stroke(width=5),
            FadeIn(split_one),
            Write(eq_one),
        )
        self.play(
            hidden[0].animate.set_color(RELEVANCE_COLOR).scale(1.15),
            hidden[1].animate.set_color(RELEVANCE_COLOR).scale(1.15),
            FadeOut(output_value),
            FadeIn(Text("0.55", font_size=20).move_to(hidden[0])),
            FadeIn(Text("0.37", font_size=20).move_to(hidden[1])),
        )

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "Each layer redistributes only the",
                    "relevance it received.",
                    "Nothing is created or destroyed.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        split_two = VGroup(
            Text("0.30", font_size=18, color=RELEVANCE_COLOR).move_to(edges[2].get_center() + UP * 0.2),
            Text("0.25", font_size=18, color=RELEVANCE_COLOR).move_to(edges[3].get_center() + DOWN * 0.2),
            Text("0.22", font_size=18, color=RELEVANCE_COLOR).move_to(edges[4].get_center() + UP * 0.2),
            Text("0.15", font_size=18, color=RELEVANCE_COLOR).move_to(edges[5].get_center() + DOWN * 0.2),
        )
        eq_two = self.make_formula(
            "0.55 = 0.30 + 0.25    0.37 = 0.22 + 0.15",
            font_size=20,
            color=SUCCESS_COLOR,
        )
        eq_two.next_to(eq_one, UP, buff=0.25)
        self.play(
            *[edges[index].animate.set_color(RELEVANCE_COLOR).set_stroke(width=4) for index in range(2, 6)],
            FadeIn(split_two),
            Write(eq_two),
        )
        input_values = VGroup(
            Text("0.30", font_size=18).move_to(inputs[0]),
            Text("0.25", font_size=18).move_to(inputs[1]),
            Text("0.22", font_size=18).move_to(inputs[2]),
            Text("0.15", font_size=18).move_to(inputs[3]),
        )
        self.play(FadeIn(input_values), inputs.animate.set_color(RELEVANCE_COLOR))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "Conservation is the key promise of",
                    "LRP: the explanation scores add",
                    "back up to the original evidence.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class GNNLRPRelevantWalkScene(TeachingScene):
    """Stage 7: walk-based explanation in a graph setting."""

    def construct(self):
        header = self.make_header(
            "Stage 7",
            "Relevant walks",
            "GNN-LRP can attribute a prediction to a multi-hop chain of interacting residues.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "What does a graph explanation look",
                    "like after relevance is pushed",
                    "back to the input graph?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        positions = {
            "r12": np.array([-4.8, 0.6, 0.0]),
            "r18": np.array([-3.6, 2.0, 0.0]),
            "r35": np.array([-2.0, 1.0, 0.0]),
            "r52": np.array([-1.2, -0.7, 0.0]),
            "r68": np.array([-3.2, -1.8, 0.0]),
            "r81": np.array([0.3, 0.4, 0.0]),
        }
        graph_nodes = VGroup(
            *[self.make_labeled_node(label, point, label_direction=DOWN, label_size=18) for label, point in positions.items()]
        )
        graph_edges = VGroup(
            Line(positions["r12"], positions["r18"], color=EDGE_COLOR, stroke_width=3),
            Line(positions["r18"], positions["r35"], color=EDGE_COLOR, stroke_width=3),
            Line(positions["r35"], positions["r52"], color=EDGE_COLOR, stroke_width=3),
            Line(positions["r52"], positions["r68"], color=EDGE_COLOR, stroke_width=3),
            Line(positions["r35"], positions["r81"], color=EDGE_COLOR, stroke_width=3),
            Line(positions["r18"], positions["r68"], color=EDGE_COLOR, stroke_width=3),
        )
        self.play(FadeIn(graph_nodes), Create(graph_edges))

        score_box = self.make_panel("Prediction", ["enzyme class", "score: 0.92"], accent=RELEVANCE_COLOR)
        score_box.scale(0.82)
        score_box.to_corner(UR, buff=0.45).shift(DOWN * 1.8)
        score_arrow = Arrow(score_box.get_left(), positions["r81"] + RIGHT * 0.35, color=EDGE_SOFT, buff=0.12, max_tip_length_to_length_ratio=0.12)
        self.play(FadeIn(score_box), Create(score_arrow))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Start from the model output and",
                    "propagate relevance backward into",
                    "the input graph.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        primary_walk = VGroup(
            graph_nodes[1][0].copy().set_color(RELEVANCE_COLOR).scale(1.35),
            graph_nodes[2][0].copy().set_color(RELEVANCE_COLOR).scale(1.35),
            graph_nodes[3][0].copy().set_color(RELEVANCE_COLOR).scale(1.35),
            Line(positions["r18"], positions["r35"], color=RELEVANCE_COLOR, stroke_width=6),
            Line(positions["r35"], positions["r52"], color=RELEVANCE_COLOR, stroke_width=6),
        )
        secondary_walk = VGroup(
            graph_nodes[5][0].copy().set_color(SECONDARY_RELEVANCE).scale(1.2),
            Line(positions["r35"], positions["r81"], color=SECONDARY_RELEVANCE, stroke_width=4),
        )
        self.play(FadeIn(primary_walk), run_time=1.0)
        self.play(FadeIn(secondary_walk), run_time=0.6)

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "The strongest explanation is a",
                    "connected walk through residues.",
                    "That captures cooperative evidence,",
                    "not only isolated importance.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        walk_label = self.make_panel(
            "Relevant walk",
            ["r18 -> r35 -> r52", "jointly supports the score"],
            accent=SUCCESS_COLOR,
        )
        walk_label.scale(0.82)
        walk_label.next_to(primary_walk, DOWN, buff=0.55)
        self.play(FadeIn(walk_label))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "GNN-LRP explains which interaction",
                    "chain mattered for the actual",
                    "prediction, not just which single",
                    "residue had a large sensitivity.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()


class WildtypeMutantDeltaScene(TeachingScene):
    """Stage 8: compare explanation shifts between wild type and mutant."""

    def construct(self):
        header = self.make_header(
            "Stage 8",
            "Wild type vs mutant",
            "The final question is how the explanation changes after a residue substitution.",
        )
        info = self.place_panel(
            self.make_panel(
                "Question",
                [
                    "How do we compare mechanism, not",
                    "just score, between two proteins?",
                ],
            ),
            UR,
            DOWN * 0.45,
        )
        self.add(header)
        self.play(FadeIn(info, shift=LEFT))

        base_positions = [
            np.array([-5.3, 0.4, 0.0]),
            np.array([-4.1, 1.8, 0.0]),
            np.array([-2.6, 0.8, 0.0]),
            np.array([-2.1, -1.0, 0.0]),
            np.array([-3.8, -1.8, 0.0]),
        ]
        mutation_index = 2

        wt_nodes = VGroup(*[Dot(point, radius=0.18, color=NODE_COLOR) for point in base_positions])
        wt_edges = VGroup(
            Line(base_positions[0], base_positions[1], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[1], base_positions[2], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[2], base_positions[3], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[3], base_positions[4], color=EDGE_COLOR, stroke_width=3),
            Line(base_positions[4], base_positions[0], color=EDGE_COLOR, stroke_width=3),
        )

        shift = RIGHT * 6.2
        mut_nodes = wt_nodes.copy().shift(shift)
        mut_edges = wt_edges.copy().shift(shift)

        wt_title = Text("wild type: E104", font_size=24, color=SUCCESS_COLOR).next_to(wt_nodes, UP, buff=0.35)
        mut_title = Text("mutant: K104", font_size=24, color=RELEVANCE_COLOR).next_to(mut_nodes, UP, buff=0.35)
        self.play(FadeIn(wt_nodes, wt_edges, mut_nodes, mut_edges), Write(wt_title), Write(mut_title))

        step_one = self.place_panel(
            self.make_panel(
                "Step 1",
                [
                    "Run the same trained model on the",
                    "wild type and the mutant graph.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_one))

        wt_mutation = wt_nodes[mutation_index].copy().set_color(SUCCESS_COLOR).scale(1.35)
        mut_mutation = mut_nodes[mutation_index].copy().set_color(RELEVANCE_COLOR).scale(1.35)
        self.play(FadeIn(wt_mutation), FadeIn(mut_mutation))

        wt_path = VGroup(
            Line(wt_nodes[1].get_center(), wt_nodes[2].get_center(), color=SUCCESS_COLOR, stroke_width=6),
            Line(wt_nodes[2].get_center(), wt_nodes[3].get_center(), color=SUCCESS_COLOR, stroke_width=6),
        )
        mut_path = VGroup(
            Line(mut_nodes[0].get_center(), mut_nodes[2].get_center(), color=SECONDARY_RELEVANCE, stroke_width=5),
            Line(mut_nodes[2].get_center(), mut_nodes[4].get_center(), color=SECONDARY_RELEVANCE, stroke_width=5),
        )
        lost_contact = DashedLine(mut_nodes[1].get_center(), mut_nodes[2].get_center(), color=RELEVANCE_COLOR, dash_length=0.12, stroke_width=3)
        lost_label = Text("relevance moved away", font_size=18, color=RELEVANCE_COLOR).next_to(lost_contact, RIGHT, buff=0.1)
        self.play(FadeIn(wt_path), FadeIn(mut_path), Create(lost_contact), FadeIn(lost_label))

        step_two = self.place_panel(
            self.make_panel(
                "Step 2",
                [
                    "Compare the relevance maps.",
                    "The mutation can reroute which",
                    "contacts support the prediction.",
                ],
                accent=NODE_ALT_COLOR,
            ),
            DR,
            LEFT * 0.2,
        )
        self.play(Transform(info, step_two))

        delta_arrow = Arrow(wt_nodes.get_right(), mut_nodes.get_left(), color=ACCENT_COLOR, buff=0.5, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        delta_panel = self.make_panel(
            "Delta insight",
            [
                "wild type keeps the original",
                "interaction path near residue 104",
                "while the mutant shifts evidence",
                "to a different neighborhood",
            ],
            accent=ACCENT_COLOR,
        )
        delta_panel.scale(0.8)
        delta_panel.move_to(np.array([0.6, -2.4, 0.0]))
        self.play(Create(delta_arrow), FadeIn(delta_panel))

        takeaway = self.place_panel(
            self.make_takeaway(
                [
                    "Case studies become mechanistic",
                    "when we compare explanation shifts,",
                    "not only final prediction scores.",
                ]
            ),
            DL,
            UP * 0.15 + RIGHT * 0.1,
        )
        self.play(Transform(info, takeaway))
        self.wait(0.9)
        self.fade_all()
