"""
30-second educational manim animation: how data flows through a neural
network. The architecture is an extended ProtCast-style binary classifier:
343 -> 64 (ReLU) -> 32 (ReLU) -> 16 (ReLU) -> 1 (Sigmoid).

At any moment only two matrices are on screen — the current representation
on the left and its next representation on the right — connected by a
labeled arrow showing the transformation. Between beats the right matrix
slides left and a new right matrix appears, so each layer's transformation
gets the screen to itself.

Render:
    pip install manim   # plus a working ffmpeg
    manim -pqh scripts/manim_dataflow.py ProtCastDataFlow      # 1080p
    manim -pql scripts/manim_dataflow.py ProtCastDataFlow      # 480p preview

Voice-over script (~30s, ~75 words):

  Beat 1 (0:00-0:04) — "Six protein feature vectors. Each has 343 numbers.
                       That is our input matrix."

  Beat 2 (0:04-0:09) — "Layer one: a dense projection from 343 features to
                       64. ReLU clips the negatives."

  Beat 3 (0:09-0:14) — "Layer two compresses again, to 32 features."

  Beat 4 (0:14-0:19) — "Layer three reduces further, to just 16."

  Beat 5 (0:19-0:25) — "The final layer collapses to a single number per
                       protein. Sigmoid turns it into a probability."

  Beat 6 (0:25-0:30) — "Each layer produces a new representation of the data."
"""

from manim import (
    Scene,
    Text,
    VGroup,
    Square,
    Circle,
    Line,
    Arrow,
    Write,
    Create,
    Indicate,
    FadeIn,
    FadeOut,
    GrowArrow,
    LaggedStart,
    interpolate_color,
    DOWN,
    UP,
    LEFT,
    RIGHT,
    UR,
    BLACK,
    WHITE,
    YELLOW,
    ORANGE,
    BLUE_B,
    BLUE_C,
    GREEN_B,
    TEAL_B,
    RED_C,
    GREY_B,
    GREY_C,
    GREY_E,
)
import numpy as np


FONT = "Helvetica"  # macOS sans-serif; Pango falls back if unavailable


def make_heatmap(values, cell_size=0.5):
    """Build a heatmap as a VGroup of colored Squares from a 2D array.

    Values are normalized to [-1, 1]; positives are red, negatives are blue,
    zero is dark grey.
    """
    values = np.asarray(values, dtype=float)
    rows, cols = values.shape
    vmax = max(abs(values.min()), abs(values.max()), 1e-9)
    normed = values / vmax
    cells = VGroup()
    for i in range(rows):
        for j in range(cols):
            v = float(normed[i, j])
            if v >= 0:
                color = interpolate_color(GREY_E, RED_C, v)
            else:
                color = interpolate_color(GREY_E, BLUE_C, -v)
            cell = Square(side_length=cell_size, stroke_width=0.5, stroke_color=BLACK)
            cell.set_fill(color, opacity=1.0)
            cell.move_to([j * cell_size, -i * cell_size, 0])
            cells.add(cell)
    cells.move_to([0, 0, 0])
    return cells


def _interpret(prob):
    """Plain-language reading of a binary-classification probability."""
    if prob >= 0.85:
        return "Yes, high probability"
    if prob >= 0.5:
        return "likely positive"
    if prob >= 0.15:
        return "likely negative"
    return "No, low probability"


def make_value_labels(matrix, values, font_size=10):
    """Tiny per-cell floating-point labels positioned at each cell's center."""
    rows, cols = values.shape
    labels = VGroup()
    for i in range(rows):
        for j in range(cols):
            cell = matrix[i * cols + j]
            t = Text(f"{values[i, j]:.2f}", font_size=font_size, color=WHITE, font=FONT)
            t.move_to(cell.get_center())
            labels.add(t)
    return labels


def make_split_heatmap(left_values, right_values, cell_size=0.5, dots_font_size=18):
    """Heatmap whose rows are split into two halves separated by '...'.

    Use this when the underlying matrix has many more columns than we want
    to show — the ellipsis signals "more cells continue here." Returns a
    VGroup of (left_mat, right_mat, ellipses), where left_mat/right_mat are
    plain heatmaps so they can still be indexed cell-by-cell.
    """
    left_mat = make_heatmap(left_values, cell_size=cell_size)
    right_mat = make_heatmap(right_values, cell_size=cell_size)
    VGroup(left_mat, right_mat).arrange(RIGHT, buff=0.55)

    rows = left_values.shape[0]
    n_left = left_values.shape[1]
    n_right = right_values.shape[1]
    ellipses = VGroup()
    for i in range(rows):
        left_last = left_mat[i * n_left + n_left - 1]
        right_first = right_mat[i * n_right]
        mid_x = (left_last.get_right()[0] + right_first.get_left()[0]) / 2
        y = left_last.get_center()[1]
        dots = Text("...", font_size=dots_font_size, color=GREY_B, font=FONT).move_to(
            [mid_x, y, 0]
        )
        ellipses.add(dots)
    return VGroup(left_mat, right_mat, ellipses)


def make_layer_column(n_visible, color, radius=0.22, v_spacing=0.7):
    """Vertical column of colored circles — a single network layer.
    Geometry matches ProtCastNetwork exactly (same radius, spacing, stroke).
    Used in DataFlow positioned on the arrow between two matrix panels."""
    total_h = (n_visible - 1) * v_spacing
    circles = VGroup(
        *[
            Circle(radius=radius, color=BLACK, stroke_width=1.5)
            .set_fill(color, opacity=1.0)
            .move_to([0, total_h / 2 - i * v_spacing, 0])
            for i in range(n_visible)
        ]
    )
    circles.set_z_index(10)  # render on top of the arrow
    return circles


def make_legend():
    """Small color key: low number (grey) -> high number (red)."""
    n = 7
    cells = VGroup()
    for i in range(n):
        v = i / (n - 1)  # 0 .. 1
        color = interpolate_color(GREY_E, RED_C, v)
        sq = Square(side_length=0.22, stroke_width=0.4, stroke_color=BLACK)
        sq.set_fill(color, opacity=1.0)
        cells.add(sq)
    cells.arrange(RIGHT, buff=0.0)
    low = (
        VGroup(
            Text("low", font_size=16, color=GREY_B, font=FONT),
            Text("number", font_size=16, color=GREY_B, font=FONT),
        )
        .arrange(DOWN, buff=0.05)
        .next_to(cells, LEFT, buff=0.15)
    )
    high = (
        VGroup(
            Text("high", font_size=16, color=GREY_B, font=FONT),
            Text("number", font_size=16, color=GREY_B, font=FONT),
        )
        .arrange(DOWN, buff=0.05)
        .next_to(cells, RIGHT, buff=0.15)
    )
    return VGroup(low, cells, high)


def make_panel(matrix, label_text, shape_text, label_color):
    """Stack a matrix above its label and shape annotation."""
    label = Text(label_text, font_size=22, weight="BOLD", color=label_color, font=FONT)
    shape = Text(shape_text, font_size=20, color=GREY_B, font=FONT)
    return VGroup(matrix, label, shape).arrange(DOWN, buff=0.2)


# (layer name, full op label, layer color, true output dim, (n_left, n_right) cells, is_sigmoid, label_color)
# Each hidden layer's visible row is split as n_left + "..." + n_right cells.
# The sigmoid output panel has a single cell, so its split is (1, 0) and is
# special-cased.
LAYERS = [
    ("Layer 1", "Dense 343 → 64, ReLU",  YELLOW,   64, (3, 3), False, YELLOW),
    ("Layer 2", "Dense 64 → 32, ReLU",   GREEN_B,  32, (3, 2), False, GREEN_B),
    ("Layer 3", "Dense 32 → 16, ReLU",   TEAL_B,   16, (2, 2), False, TEAL_B),
    ("Layer 4", "Dense 16 → 1, Sigmoid", ORANGE,    1, (1, 0), True,  ORANGE),
]

# Per-layer node counts for the compact layer columns drawn on each DataFlow
# arrow. These match the "After Layer N" / Output node counts used in
# ProtCastNetwork (see NETWORK_LAYERS).
LAYER_NODE_COUNTS = [5, 4, 3, 1]

LEFT_X = -3.6
RIGHT_X = 3.4
PANEL_Y = -0.3
CELL = 0.5


class ProtCastDataFlow(Scene):
    def construct(self):
        np.random.seed(7)

        # ===== Title  (0.0 -> 1.5s) =====
        title = Text(
            "Data Flow Through a Neural Network", font_size=30, font=FONT
        ).to_edge(UP, buff=0.4)
        subtitle = Text(
            "Inspired by ProtCast (https://github.com/bioteam/protcast)",
            font_size=18,
            color=GREY_B,
            font=FONT,
        ).next_to(title, DOWN, buff=0.1)
        legend = make_legend().to_corner(UR, buff=0.35)
        self.play(
            Write(title),
            FadeIn(subtitle, shift=DOWN * 0.15),
            FadeIn(legend),
            run_time=1.5,
        )
        self.wait(1.5)

        # ===== Beat 1: Input panel  (1.5 -> 4.0s) =====
        # One protein at a time. We share the seed with ProtCastPreprocessing,
        # so this row is identical to Protein 1's row in that scene.
        full_input = np.random.rand(6, 8)
        input_row = full_input[0:1, :]  # 1 x 8
        input_n_left = 4
        input_mat = make_split_heatmap(
            input_row[:, :input_n_left], input_row[:, input_n_left:], cell_size=CELL
        )
        input_panel = make_panel(input_mat, "Input", "(1, 343)", BLUE_B)
        input_panel.move_to([LEFT_X, PANEL_Y, 0])
        self.play(FadeIn(input_panel, scale=0.95), run_time=1.5)
        self.wait(5.5)

        # Pick one protein's logit/probability to flow through.
        final_prob = float(1.0 / (1.0 + np.exp(-1.8)))  # = sigmoid(1.8) ≈ 0.86

        relu_desc = None  # bottom-of-frame caption explaining ReLU
        sigmoid_desc = None  # bottom-of-frame caption explaining Sigmoid

        left_panel = input_panel
        for i, (name, op, color, out_dim, vis_split, is_sig, label_color) in enumerate(LAYERS):
            n_left, n_right = vis_split

            # ReLU caption is no longer relevant when we hit the sigmoid layer.
            if is_sig and relu_desc is not None:
                self.play(FadeOut(relu_desc), run_time=0.6)
                self.wait(0.5)
                relu_desc = None

            # Build the right-side panel (one row, possibly split with '...').
            if is_sig:
                data = np.array([[final_prob]])
                right_mat = make_heatmap(data, cell_size=CELL)
                right_label = "Output"
            else:
                pre_left = np.random.randn(1, n_left) * 0.6
                pre_right = np.random.randn(1, n_right) * 0.6
                right_mat = make_split_heatmap(
                    np.maximum(pre_left, 0.0),
                    np.maximum(pre_right, 0.0),
                    cell_size=CELL,
                )
                right_label = f"After {name}"

            right_panel = make_panel(
                right_mat, right_label, f"(1, {out_dim})", label_color
            )
            # Output panel sits further left than the hidden panels so there's
            # room to the right for the probability + interpretation label.
            right_x = 1.6 if is_sig else RIGHT_X
            right_panel.move_to([right_x, PANEL_Y, 0])

            # Arrow + compact layer column + caption between panels.
            arrow = Arrow(
                start=left_panel[0].get_right() + RIGHT * 0.18,
                end=right_panel[0].get_left() + LEFT * 0.18,
                color=color,
                buff=0.0,
                stroke_width=5,
            )
            layer_col = make_layer_column(LAYER_NODE_COUNTS[i], color)
            layer_col.move_to((arrow.get_start() + arrow.get_end()) / 2)
            op_top = Text(name, font_size=22, weight="BOLD", color=color, font=FONT).next_to(
                layer_col, UP, buff=0.25
            )
            op_bot = Text(op, font_size=18, color=color, font=FONT).next_to(
                layer_col, DOWN, buff=0.25
            )

            self.play(
                GrowArrow(arrow),
                FadeIn(layer_col, scale=0.9),
                Write(op_top),
                Write(op_bot),
                run_time=1.5,
            )
            self.wait(1.5)
            self.play(FadeIn(right_panel, shift=RIGHT * 0.3), run_time=1.5)
            self.wait(2.0)

            # Briefly reveal each layer's W matrix so the viewer sees that
            # every layer has its own learned weights, with shapes that change
            # layer by layer.
            prev_dim = 343 if i == 0 else LAYERS[i - 1][3]
            W_data = np.random.rand(4, 3)
            W_mat = make_heatmap(W_data, cell_size=0.18)
            W_caption = Text(
                f"Learned weights W: ({prev_dim} × {out_dim})",
                font_size=18,
                color=GREY_B,
                font=FONT,
            )
            W_group = VGroup(W_mat, W_caption).arrange(DOWN, buff=0.12)
            arrow_mid = (arrow.get_start() + arrow.get_end()) / 2
            W_group.move_to([arrow_mid[0], -2.7, 0])
            self.play(FadeIn(W_group), run_time=1.0)
            self.wait(3.0)
            self.play(FadeOut(W_group), run_time=0.7)
            self.wait(1.0)

            # After the first layer's panel is in place, introduce ReLU.
            if relu_desc is None and not is_sig:
                relu_desc = Text(
                    "ReLU: f(x) = max(0, x). Positives pass through; negatives become zero.",
                    font_size=18,
                    color=GREY_B,
                    font=FONT,
                ).to_edge(DOWN, buff=0.5)
                self.play(Write(relu_desc), run_time=1.5)
                self.wait(3.0)

            if is_sig:
                prob_text = Text(
                    f"{final_prob:.2f}", font_size=20, color=WHITE, font=FONT
                ).next_to(right_mat[0], RIGHT, buff=0.18)
                interp_text = Text(
                    f"({_interpret(final_prob)})",
                    font_size=20,
                    color=GREY_B,
                    font=FONT,
                ).next_to(prob_text, RIGHT, buff=0.2)
                self.play(Write(VGroup(prob_text, interp_text)), run_time=1.5)
                self.wait(2.5)

                sigmoid_desc = Text(
                    "Sigmoid: σ(x) = 1 / (1 + e⁻ˣ). Maps any real number to a probability in [0, 1].",
                    font_size=18,
                    color=GREY_B,
                    font=FONT,
                ).to_edge(DOWN, buff=0.5)
                self.play(Write(sigmoid_desc), run_time=1.5)
                self.wait(3.0)

            self.wait(7.0)

            # Slide pipeline left for the next layer (skip after the last layer).
            if i < len(LAYERS) - 1:
                self.play(
                    FadeOut(left_panel, shift=LEFT * 0.5),
                    FadeOut(arrow),
                    FadeOut(layer_col),
                    FadeOut(op_top),
                    FadeOut(op_bot),
                    right_panel.animate.move_to([LEFT_X, PANEL_Y, 0]),
                    run_time=1.5,
                )
                self.wait(0.5)
                left_panel = right_panel

        # ===== Outro =====
        if sigmoid_desc is not None:
            self.play(FadeOut(sigmoid_desc), run_time=0.6)
            self.wait(0.5)
        outro = Text(
            "Each hidden layer creates a new representation of the data — applying its formula to learned weights.",
            font_size=20,
            color=YELLOW,
            font=FONT,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(outro), run_time=2.0)
        self.wait(9.0)


# Network architecture (matches the matrix-flow video).
# (visible nodes, true dim, layer color)
NETWORK_LAYERS = [
    (6, 343, BLUE_B),    # Input
    (5,  64, YELLOW),    # After Layer 1
    (4,  32, GREEN_B),   # After Layer 2
    (3,  16, TEAL_B),    # After Layer 3
    (1,   1, ORANGE),    # Output
]


class ProtCastNetwork(Scene):
    """Classical node-and-edge neural network diagram.

    Renders ~15s. Intended to be played BEFORE ProtCastDataFlow so the viewer
    first sees the network's structure (nodes & connections), then the matrix
    view of how data flows through it.

    Render both back-to-back:
        manim -pqh scripts/manim_dataflow.py ProtCastNetwork ProtCastDataFlow
    Then concatenate the two MP4s with ffmpeg if you want a single file:
        ffmpeg -i ProtCastNetwork.mp4 -i ProtCastDataFlow.mp4 \\
               -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0" combined.mp4
    """

    def construct(self):
        node_radius = 0.22
        v_spacing = 0.7
        x_positions = [-4.4, -2.2, 0.0, 2.2, 4.4]

        # ===== Title  (0.0 -> 1.2s) =====
        title = Text(
            "Neural Network Architecture", font_size=30, font=FONT
        ).to_edge(UP, buff=0.4)
        subtitle = Text(
            "Inspired by ProtCast (https://github.com/bioteam/protcast)",
            font_size=18,
            color=GREY_B,
            font=FONT,
        ).next_to(title, DOWN, buff=0.1)
        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.15), run_time=1.2)

        # Build all node columns and dim labels.
        all_layers = []
        all_dim_labels = []
        for (n_visible, true_dim, color), x in zip(NETWORK_LAYERS, x_positions):
            total_h = (n_visible - 1) * v_spacing
            nodes = VGroup(
                *[
                    Circle(radius=node_radius, color=BLACK, stroke_width=1.5)
                    .set_fill(color, opacity=1.0)
                    .move_to([x, total_h / 2 - i * v_spacing, 0])
                    for i in range(n_visible)
                ]
            )
            dim_text = Text(f"{true_dim}", font_size=20, color=GREY_B, font=FONT)
            dim_text.next_to(nodes, DOWN, buff=0.35)
            all_layers.append(nodes)
            all_dim_labels.append(dim_text)

        neurons_label = Text("neurons:", font_size=18, color=GREY_B, font=FONT)
        neurons_label.next_to(all_dim_labels[0], LEFT, buff=0.35)

        # Build connection lines (full mesh between adjacent layers).
        connections_list = []
        for i in range(len(all_layers) - 1):
            prev_nodes = all_layers[i]
            curr_nodes = all_layers[i + 1]
            conns = VGroup()
            for src in prev_nodes:
                for dst in curr_nodes:
                    line = Line(
                        src.get_center() + RIGHT * node_radius,
                        dst.get_center() + LEFT * node_radius,
                        color=GREY_C,
                        stroke_width=1.2,
                        stroke_opacity=0.55,
                    )
                    conns.add(line)
            connections_list.append(conns)

        # Section labels above each region of the diagram.
        section_y = 2.55
        input_section = Text(
            "Input layer matches input data",
            font_size=22,
            weight="BOLD",
            color=BLUE_B,
            font=FONT,
        ).move_to([x_positions[0], section_y, 0])
        hidden_section = Text(
            "Hidden layers", font_size=22, weight="BOLD", color=GREY_B, font=FONT
        ).move_to([x_positions[2], section_y, 0])  # centered above the 3 hidden columns
        output_section = Text(
            "Output layer matches prediction",
            font_size=22,
            weight="BOLD",
            color=ORANGE,
            font=FONT,
        ).move_to([x_positions[-1], section_y, 0])

        # ===== Beat 1: Input layer + arrows  (1.2 -> 3.0s) =====
        input_arrows = VGroup(
            *[
                Arrow(
                    start=node.get_center() + LEFT * 0.85,
                    end=node.get_center() + LEFT * (node_radius + 0.04),
                    color=GREY_B,
                    buff=0.0,
                    stroke_width=2.5,
                    max_tip_length_to_length_ratio=0.25,
                )
                for node in all_layers[0]
            ]
        )
        self.play(
            FadeIn(all_layers[0], scale=0.95),
            FadeIn(all_dim_labels[0], shift=DOWN * 0.15),
            FadeIn(neurons_label, shift=DOWN * 0.15),
            FadeIn(input_section, shift=DOWN * 0.15),
            *[GrowArrow(a) for a in input_arrows],
            run_time=1.4,
        )
        self.wait(0.4)

        # ===== Beats 2-5: Each subsequent layer fans in  (3.0 -> 9.8s) =====
        for i in range(1, len(all_layers)):
            extras = []
            if i == 1:
                extras.append(FadeIn(hidden_section, shift=DOWN * 0.15))
            elif i == len(all_layers) - 1:
                extras.append(FadeIn(output_section, shift=DOWN * 0.15))
            self.play(
                Create(connections_list[i - 1], lag_ratio=0.01),
                FadeIn(all_layers[i], scale=0.95),
                FadeIn(all_dim_labels[i], shift=DOWN * 0.15),
                *extras,
                run_time=1.4,
            )
            self.wait(0.3)

        # ===== Beat 6: Output arrow  (9.8 -> 10.5s) =====
        out_node = all_layers[-1][0]
        output_arrow = Arrow(
            start=out_node.get_center() + RIGHT * (node_radius + 0.04),
            end=out_node.get_center() + RIGHT * 0.9,
            color=GREY_B,
            buff=0.0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.25,
        )
        prediction_label = Text(
            "Prediction", font_size=18, color=GREY_B, font=FONT
        ).next_to(output_arrow, RIGHT, buff=0.15)
        self.play(GrowArrow(output_arrow), Write(prediction_label), run_time=0.7)

        hidden_desc = Text(
            "Hidden layers: y = ReLU(Wx + b). "
            "Each line in the diagram is one weight in W, learned from data.",
            font_size=18,
            color=GREY_B,
            font=FONT,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(hidden_desc), run_time=1.2)
        self.wait(3.4)

        # ===== Beat 7: Signal pulse — data propagates left to right  (10.5 -> 14.5s) =====
        self.play(Indicate(all_layers[0], color=YELLOW, scale_factor=1.18), run_time=0.4)
        for i in range(len(connections_list)):
            self.play(
                connections_list[i].animate.set_stroke(YELLOW, width=2.0, opacity=0.95),
                run_time=0.3,
            )
            self.play(
                connections_list[i].animate.set_stroke(GREY_C, width=1.2, opacity=0.55),
                Indicate(all_layers[i + 1], color=YELLOW, scale_factor=1.18),
                run_time=0.4,
            )

        # ===== Outro  (~14.5 -> 15.5s) =====
        self.wait(1.0)


# Six placeholder protein sequences. Each starts with M (methionine, the
# standard start codon) and ends with "..." to suggest indeterminate length.
# Lengths are deliberately varied to show that real proteins differ in size.
EXAMPLE_SEQUENCES = [
    "MKVLWAALLVTFLAGCQAKVEPSRTGAH...",
    "MSGGFKLLAV...",
    "MQTLPVRRYGLAVALLSAACGFRSTPNL...",
    "MFAAGLA...",
    "MSTRYRVQLALMSGAFAA...",
    "MIVELRDPLAGAVAATSGAALPQNTV...",
]


class ProtCastPreprocessing(Scene):
    """6 protein sequences -> 6x343 input matrix via CTriad encoding.

    ~24s. Intended to play FIRST in the trilogy:
        ProtCastPreprocessing -> ProtCastNetwork -> ProtCastDataFlow

    Render the trilogy:
        manim -pqh scripts/manim_dataflow.py \\
            ProtCastPreprocessing ProtCastNetwork ProtCastDataFlow

    Voice-over (~24s, ~55 words):
      Beat 1 (0:00-0:02) — "Start with six protein sequences."
      Beat 2 (0:02-0:08) — "Each is a string of amino acids."
      Beat 3 (0:08-0:13) — "We encode every sequence using CTriad —
                          counting groups of three across seven physico-
                          chemical classes. Seven cubed gives 343 features."
      Beat 4 (0:13-0:20) — "Each protein becomes one row of numbers."
      Beat 5 (0:20-0:24) — "Together, they form the input matrix:
                           six proteins by 343 features."
    """

    def construct(self):
        # Seed matches ProtCastDataFlow so the input matrix is identical
        # across both scenes (same 6 proteins -> same feature values).
        np.random.seed(7)

        # ===== Title (0.0 -> 1.5s) =====
        title = Text(
            "Preprocessing: Sequences → Feature Vectors",
            font_size=26,
            font=FONT,
        ).to_edge(UP, buff=0.4)
        subtitle = Text(
            "CTriad encoding (7³ = 343 features per protein)",
            font_size=20,
            color=GREY_B,
            font=FONT,
        ).next_to(title, DOWN, buff=0.1)
        legend = make_legend().to_corner(UR, buff=0.35)
        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.15), run_time=1.2)

        # ===== Beat 1: 6 sequences appear  (1.5 -> 7.5s) =====
        seq_rows = VGroup()
        for i, seq in enumerate(EXAMPLE_SEQUENCES):
            label = Text(f"Protein {i + 1}:", font_size=20, color=GREY_B, font=FONT)
            seq_text = Text(seq, font_size=22, font="Menlo", color=WHITE)
            row = VGroup(label, seq_text).arrange(RIGHT, buff=0.3)
            seq_rows.add(row)
        seq_rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        seq_rows.move_to([0, 0.2, 0])

        self.play(
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.3) for row in seq_rows],
                lag_ratio=0.1,
            ),
            run_time=2.0,
        )
        self.wait(3.0)

        # ===== Beat 2: Slide sequences left, show encoding arrow  (7.5 -> 11.5s) =====
        self.play(
            seq_rows.animate.scale(0.85).move_to([-3.7, 0.2, 0]),
            FadeIn(legend),
            run_time=1.2,
        )

        encode_arrow = Arrow(
            start=[0.0, 0.2, 0],
            end=[1.3, 0.2, 0],
            color=YELLOW,
            buff=0.0,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.22,
        )
        encode_top = Text(
            "CTriad", font_size=22, weight="BOLD", color=YELLOW, font=FONT
        ).next_to(encode_arrow, UP, buff=0.18)
        encode_bot = Text(
            "encode", font_size=18, color=YELLOW, font=FONT
        ).next_to(encode_arrow, DOWN, buff=0.18)
        self.play(GrowArrow(encode_arrow), Write(encode_top), Write(encode_bot), run_time=1.2)

        ctriad_desc = Text(
            "CTriad represents each protein as counts of 343 triplets across 7 amino-acid groups.",
            font_size=18,
            color=GREY_B,
            font=FONT,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(ctriad_desc), run_time=1.5)
        self.wait(0.4)

        # ===== Beat 3: Build matrix one row at a time  (11.5 -> 18.5s) =====
        # Show matrix as two halves with "..." between them, signaling that the
        # full matrix is much wider than 8 columns (343 in reality).
        n_left, n_right = 4, 4
        feature_data = np.random.rand(6, n_left + n_right)  # CTriad counts are non-negative; floats in [0, 1]
        left_data = feature_data[:, :n_left]
        right_data = feature_data[:, n_left:]
        left_mat = make_heatmap(left_data, cell_size=0.5)
        right_mat = make_heatmap(right_data, cell_size=0.5)
        VGroup(left_mat, right_mat).arrange(RIGHT, buff=0.55).move_to([4.2, 0.2, 0])
        left_labels = make_value_labels(left_mat, left_data, font_size=11)
        right_labels = make_value_labels(right_mat, right_data, font_size=11)

        # Per-row "..." in the gap between the two halves.
        ellipses = VGroup()
        for i in range(6):
            left_last = left_mat[i * n_left + n_left - 1]
            right_first = right_mat[i * n_right]
            mid_x = (left_last.get_right()[0] + right_first.get_left()[0]) / 2
            y = left_last.get_center()[1]
            dots = Text("...", font_size=18, color=GREY_B, font=FONT).move_to([mid_x, y, 0])
            ellipses.add(dots)

        rows = [
            VGroup(
                *[left_mat[i * n_left + j] for j in range(n_left)],
                *[right_mat[i * n_right + j] for j in range(n_right)],
                *[left_labels[i * n_left + j] for j in range(n_left)],
                *[right_labels[i * n_right + j] for j in range(n_right)],
                ellipses[i],
            )
            for i in range(6)
        ]

        for i in range(6):
            self.play(
                Indicate(seq_rows[i], color=YELLOW, scale_factor=1.0),
                FadeIn(rows[i], shift=LEFT * 0.3),
                run_time=0.8,
            )

        matrix_label = Text(
            "Input data", font_size=20, weight="BOLD", color=BLUE_B, font=FONT
        )
        matrix_shape = Text("(6, 343)", font_size=20, color=GREY_B, font=FONT)
        labels = VGroup(matrix_label, matrix_shape).arrange(DOWN, buff=0.1)
        labels.next_to(VGroup(left_mat, right_mat), DOWN, buff=0.3)
        self.play(Write(matrix_label), Write(matrix_shape), run_time=1.0)
        self.wait(1.5)

        # ===== Beat 4: Outro  (18.5 -> 24s) =====
        self.play(FadeOut(ctriad_desc), run_time=0.4)
        outro = Text(
            "These are the inputs to the neural network.",
            font_size=22,
            color=YELLOW,
            font=FONT,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(outro), run_time=1.5)
        self.wait(3.0)


class ProtCastIntro(Scene):
    """Intro: frames the problem ProtCast solves — binary classification of
    protein sequences. ~28s. Plays first in the four-scene sequence:

        ProtCastIntro -> ProtCastPreprocessing -> ProtCastNetwork -> ProtCastDataFlow

    Render the full set:
        manim -pqh scripts/manim_dataflow.py \\
            ProtCastIntro ProtCastPreprocessing ProtCastNetwork ProtCastDataFlow

    Voice-over (~15s, ~40 words):
      Beat 1 (0:00-0:02) — "The problem: binary classification of proteins."
      Beat 2 (0:02-0:06) — "Goal: predict the class of a protein from its
                          sequence — for example, is it a GPCR?"
      Beat 3 (0:06-0:09) — "A protein is a string of amino acids."
      Beat 4 (0:09-0:13) — "GPCR — yes or no?"
      Beat 5 (0:13-0:15) — "First, turn sequences into numbers."
    """

    def construct(self):
        # ===== Beat 1: Title (0.0 -> 2.0s) =====
        title = Text("The Problem", font_size=46, weight="BOLD", font=FONT)
        subtitle = Text(
            "Binary classification of protein sequences",
            font_size=24,
            color=GREY_B,
            font=FONT,
        )
        title_grp = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.2), run_time=1.0)
        self.wait(1.0)

        # ===== Beat 2: Goal (2.0 -> 6.0s) =====
        self.play(title_grp.animate.scale(0.55).to_edge(UP, buff=0.4), run_time=0.5)

        goal_grp = VGroup(
            Text("Goal", font_size=36, weight="BOLD", color=YELLOW, font=FONT),
            Text(
                "Predict the class of a protein, from its sequence.",
                font_size=24,
                font=FONT,
                color=WHITE,
            ),
            Text(
                "For example, is it a GPCR or not?",
                font_size=22,
                font=FONT,
                color=GREY_B,
            ),
        ).arrange(DOWN, buff=0.3).move_to([0, 0.3, 0])

        self.play(Write(goal_grp), run_time=1.7)
        self.wait(1.8)

        # ===== Beat 3: A protein appears (6.0 -> 9.0s) =====
        self.play(FadeOut(goal_grp), run_time=0.5)

        seq = "MKVLWAALLVTFLAGCQAKVEPSRTGAH..."
        seq_text = Text(seq, font_size=32, font="Menlo", color=WHITE)
        seq_label = Text(
            "an amino-acid sequence", font_size=20, color=GREY_B, font=FONT
        )
        seq_grp = VGroup(seq_text, seq_label).arrange(DOWN, buff=0.2).move_to([0, 1.4, 0])

        self.play(FadeIn(seq_text, shift=DOWN * 0.3), Write(seq_label), run_time=1.2)
        self.wait(1.3)

        # ===== Beat 4: GPCR? -> Yes / No (9.0 -> 13.0s) =====
        arrow = Arrow(
            start=[0, 0.4, 0],
            end=[0, -0.7, 0],
            color=YELLOW,
            buff=0.0,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.2,
        )
        question = Text(
            "GPCR?", font_size=26, color=YELLOW, weight="BOLD", font=FONT
        ).next_to(arrow, LEFT, buff=0.25)
        go_id = Text(
            "GO:0004930", font_size=20, color=GREY_B, font=FONT
        ).next_to(arrow, RIGHT, buff=0.25)

        yes_text = Text(
            "✓ Yes (high probability)", font_size=28, color=GREEN_B, weight="BOLD", font=FONT
        )
        no_text = Text(
            "✗ No (low probability)", font_size=28, color=GREY_B, weight="BOLD", font=FONT
        )
        decisions = VGroup(yes_text, no_text).arrange(RIGHT, buff=2.5).move_to([0, -1.8, 0])

        self.play(GrowArrow(arrow), Write(question), Write(go_id), run_time=0.8)
        self.play(
            FadeIn(yes_text, shift=DOWN * 0.2),
            FadeIn(no_text, shift=DOWN * 0.2),
            run_time=1.0,
        )
        self.wait(2.2)

        # ===== Beat 5: Outro (13.0 -> 15.0s) =====
        outro = Text(
            "How? — start by turning sequences into numbers.",
            font_size=20,
            color=GREY_B,
            font=FONT,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(outro), run_time=1.2)
        self.wait(0.8)
