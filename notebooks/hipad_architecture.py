"""HiP-AD Architecture Diagram PPT Generator"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)

# ── Colors ──
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(0, 0, 0)
GRAY_BG = RGBColor(245, 245, 245)
DARK_GRAY = RGBColor(100, 100, 100)
LIGHT_GRAY = RGBColor(200, 200, 200)

C_GRP_A = RGBColor(255, 179, 179)   # red-ish
C_GRP_A_BORDER = RGBColor(211, 47, 47)
C_GRP_B = RGBColor(179, 217, 255)   # blue-ish
C_GRP_B_BORDER = RGBColor(21, 101, 192)
C_GRP_C = RGBColor(217, 179, 255)   # purple-ish
C_GRP_C_BORDER = RGBColor(123, 31, 162)
C_GRP_D = RGBColor(255, 224, 179)   # orange-ish
C_GRP_D_BORDER = RGBColor(230, 81, 0)
C_TASK = RGBColor(200, 230, 201)     # green-ish
C_TASK_BORDER = RGBColor(56, 142, 60)
C_CONCAT = RGBColor(232, 234, 246)   # light indigo
C_MOTION = RGBColor(255, 249, 196)   # yellow-ish
C_LOSS = RGBColor(255, 205, 210)     # light red
C_NOT_MEASURED = RGBColor(224, 224, 224)  # gray
C_TITLE_BG = RGBColor(33, 33, 33)


def add_box(slide, left, top, width, height, text, fill_color, border_color=None,
            font_size=10, bold=False, font_color=BLACK, align=PP_ALIGN.CENTER,
            border_width=Pt(1)):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.color.rgb = RGBColor(180, 180, 180)
        shape.line.width = Pt(0.5)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.paragraphs[0].alignment = align
    tf.paragraphs[0].space_before = Pt(0)
    tf.paragraphs[0].space_after = Pt(0)

    run = tf.paragraphs[0].add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = font_color
    # vertical center
    tf.paragraphs[0].alignment = align
    from pptx.oxml.ns import qn
    bodyPr = tf.paragraphs[0]._p.getparent()
    if bodyPr.tag.endswith('txBody'):
        bp = bodyPr.find(qn('a:bodyPr'))
        if bp is not None:
            bp.set('anchor', 'ctr')
    return shape


def add_arrow(slide, x1, y1, x2, y2, color=BLACK, width=Pt(1.5)):
    connector = slide.shapes.add_connector(
        1, x1, y1, x2, y2  # MSO_CONNECTOR.STRAIGHT = 1
    )
    connector.line.color.rgb = color
    connector.line.width = width
    # arrowhead
    connector.end_x = x2
    connector.end_y = y2
    from pptx.oxml.ns import qn
    ln = connector.line._ln
    tail = ln.find(qn('a:tailEnd'))
    if tail is None:
        from lxml import etree
        tail = etree.SubElement(ln, qn('a:tailEnd'))
    tail.set('type', 'triangle')
    tail.set('w', 'med')
    tail.set('len', 'med')
    return connector


def add_text(slide, left, top, width, height, text, font_size=10,
             bold=False, color=BLACK, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txBox


def add_dashed_border(slide, left, top, width, height, color, label_text="", label_pos='right'):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.background()
    shape.line.color.rgb = color
    shape.line.width = Pt(2.5)
    shape.line.dash_style = 2  # dash
    # label
    if label_text:
        if label_pos == 'right':
            lx = left + width + Inches(0.1)
            ly = top + height // 2 - Inches(0.15)
        else:
            lx = left + Inches(0.1)
            ly = top - Inches(0.25)
        add_text(slide, lx, ly, Inches(1.8), Inches(0.3),
                 label_text, font_size=9, bold=True, color=color)
    return shape


# ════════════════════════════════════════════════════════════════
# Slide 1: Full Architecture Overview
# ════════════════════════════════════════════════════════════════
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
slide1.background.fill.solid()
slide1.background.fill.fore_color.rgb = WHITE

# Title
add_box(slide1, Inches(0.3), Inches(0.2), Inches(12.7), Inches(0.5),
        "HiP-AD Architecture + Gradient Conflict Measurement Groups",
        C_TITLE_BG, font_size=18, bold=True, font_color=WHITE)

# ── Layout constants ──
LM = Inches(0.5)       # left margin
COL_W = Inches(2.2)    # task column width
COL_GAP = Inches(0.3)  # gap between task columns
BOX_H = Inches(0.45)
ROW_GAP = Inches(0.15)

def col_x(i):
    return LM + i * (COL_W + COL_GAP)

def task_boxes(slide, y, texts, color, border=None, font_size=9, bold=False):
    shapes = []
    for i, t in enumerate(texts):
        s = add_box(slide, col_x(i), y, COL_W, BOX_H, t, color,
                    border_color=border, font_size=font_size, bold=bold)
        shapes.append(s)
    return shapes

# Row positions
y = Inches(1.0)

# Camera Images
add_box(slide1, Inches(2.5), y, Inches(5), Inches(0.4),
        "6x Camera Images", WHITE, BLACK, font_size=11, bold=True)
y += Inches(0.55)

# ── Group A: Backbone + Neck ──
ga_top = y - Inches(0.1)
add_box(slide1, LM, y, Inches(4.5), Inches(0.55),
        "img_backbone (ResNet-50)", C_GRP_A, C_GRP_A_BORDER, font_size=11, bold=True)
add_box(slide1, LM + Inches(5.0), y, Inches(4.5), Inches(0.55),
        "img_neck (FPN)", C_GRP_A, C_GRP_A_BORDER, font_size=11, bold=True)
# arrow backbone -> neck
add_arrow(slide1, LM + Inches(4.5), y + Inches(0.27),
          LM + Inches(5.0), y + Inches(0.27))
add_dashed_border(slide1, LM - Inches(0.1), ga_top, Inches(10.2), Inches(0.75),
                  C_GRP_A_BORDER, "Group A", 'right')
y += Inches(0.85)

# Features arrow
feat_x = LM + Inches(7.25)
add_arrow(slide1, feat_x, y - Inches(0.15), feat_x, y + Inches(0.1))
add_text(slide1, feat_x - Inches(1), y, Inches(2), Inches(0.2),
         "Multi-scale Features", font_size=8, color=DARK_GRAY, align=PP_ALIGN.CENTER)
y += Inches(0.3)

# ── Instance Banks (task-specific) ──
task_boxes(slide1, y, ["det queries (900)", "map queries (100)",
                       "plan queries", "ego queries"],
           C_TASK, C_TASK_BORDER, font_size=9)
add_text(slide1, col_x(4) - Inches(0.3), y, Inches(2.5), BOX_H,
         "Instance Banks\n(task-specific)", font_size=8, color=DARK_GRAY)
y += BOX_H + ROW_GAP

# Anchor Encoders
task_boxes(slide1, y, ["det_anchor_enc", "map_anchor_enc",
                       "plan_anchor_enc", "ego_anchor_enc"],
           C_TASK, C_TASK_BORDER, font_size=8)
y += BOX_H + ROW_GAP

# CONCAT
add_box(slide1, LM, y, Inches(9.6), Inches(0.35),
        "CONCAT  [ det | map | plan | ego ]  \u2192  [B, N_total, 256]",
        C_CONCAT, font_size=9, bold=True)
y += Inches(0.5)

# ── Decoder Cycle ──
cycle_top = y - Inches(0.05)
add_text(slide1, LM, y - Inches(0.02), Inches(3), Inches(0.2),
         "Decoder Cycle (\u00d76)", font_size=9, bold=True, color=DARK_GRAY)
y += Inches(0.2)

# temp_gnn
add_box(slide1, LM, y, Inches(5.5), BOX_H,
        "temp_gnn (TemporalSeparateAttn)", C_NOT_MEASURED, font_size=9)
add_text(slide1, LM + Inches(5.8), y, Inches(4.5), BOX_H,
         "det\u2194det, map\u2194map, plan/ego\u2194det,map\n(per-group weights, not measured)",
         font_size=7, color=DARK_GRAY)
y += BOX_H + ROW_GAP

# gnn
add_box(slide1, LM, y, Inches(5.5), BOX_H,
        "gnn (SeparateAttention)", C_NOT_MEASURED, font_size=9)
add_text(slide1, LM + Inches(5.8), y, Inches(4.5), BOX_H,
         "det self-attn, map self-attn (independent weights)\n(per-group weights, not measured)",
         font_size=7, color=DARK_GRAY)
y += BOX_H + ROW_GAP

# ── Group C: fc_before ──
gc_top = y - Inches(0.05)
add_box(slide1, LM, y, Inches(5.5), Inches(0.35),
        "fc_before  Linear(256\u2192512)", C_GRP_C, C_GRP_C_BORDER, font_size=9, bold=True)
add_text(slide1, LM + Inches(5.8), y, Inches(3), Inches(0.35),
         "decouple_attn value projection", font_size=7, color=DARK_GRAY)
y += Inches(0.45)

# ── Group D: inter_gnn ──
gd_top = y - Inches(0.05)
add_box(slide1, LM, y, Inches(5.5), BOX_H,
        "inter_gnn (InteractiveAttn)", C_GRP_D, C_GRP_D_BORDER, font_size=9, bold=True)
add_text(slide1, LM + Inches(5.8), y, Inches(4.5), BOX_H,
         "plan/ego \u2192 det/map (unidirectional cross-attention)",
         font_size=8, color=DARK_GRAY)
add_dashed_border(slide1, LM - Inches(0.05), gd_top, Inches(5.7), Inches(0.55),
                  C_GRP_D_BORDER, "Group D", 'right')
y += BOX_H + ROW_GAP

# fc_after
add_box(slide1, LM, y, Inches(5.5), Inches(0.35),
        "fc_after  Linear(512\u2192256)", C_GRP_C, C_GRP_C_BORDER, font_size=9, bold=True)
add_dashed_border(slide1, LM - Inches(0.05), gc_top, Inches(5.7), y + Inches(0.35) - gc_top + Inches(0.05),
                  C_GRP_C_BORDER, "Group C", 'right')
y += Inches(0.45)

# ── Group B: norm + ffn ──
gb_top = y - Inches(0.05)

# norm 1
add_box(slide1, LM, y, Inches(5.5), Inches(0.35),
        "norm (LayerNorm)", C_GRP_B, C_GRP_B_BORDER, font_size=9, bold=True)
add_text(slide1, LM + Inches(5.8), y, Inches(3.5), Inches(0.35),
         "all task queries concat \u2192 same LN weights", font_size=7, color=DARK_GRAY)
y += Inches(0.4)

# SPLIT
add_box(slide1, LM, y, Inches(9.6), Inches(0.3),
        "SPLIT \u2192 det / map / plan / ego", C_CONCAT, font_size=8)
y += Inches(0.35)

# deformable (task-specific)
task_boxes(slide1, y, ["det_deformable", "map_deformable",
                       "plan_deformable", "ego_deformable"],
           C_TASK, C_TASK_BORDER, font_size=8)
add_text(slide1, col_x(4) - Inches(0.3), y, Inches(2.5), BOX_H,
         "Image feat extraction\n(task-specific)", font_size=7, color=DARK_GRAY)
y += BOX_H + ROW_GAP

# CONCAT 2
add_box(slide1, LM, y, Inches(9.6), Inches(0.3),
        "CONCAT", C_CONCAT, font_size=8)
y += Inches(0.35)

# ffn
add_box(slide1, LM, y, Inches(5.5), Inches(0.35),
        "ffn (AsymmetricFFN)", C_GRP_B, C_GRP_B_BORDER, font_size=9, bold=True)
add_text(slide1, LM + Inches(5.8), y, Inches(3.5), Inches(0.35),
         "all task queries concat \u2192 same FFN weights", font_size=7, color=DARK_GRAY)
y += Inches(0.4)

# norm 2
add_box(slide1, LM, y, Inches(5.5), Inches(0.35),
        "norm (LayerNorm)", C_GRP_B, C_GRP_B_BORDER, font_size=9, bold=True)
add_dashed_border(slide1, LM - Inches(0.05), gb_top, Inches(5.7), y + Inches(0.35) - gb_top + Inches(0.05),
                  C_GRP_B_BORDER, "Group B", 'right')
y += Inches(0.4)

# SPLIT 2
add_box(slide1, LM, y, Inches(9.6), Inches(0.3),
        "SPLIT \u2192 det / map / plan / ego", C_CONCAT, font_size=8)
y += Inches(0.35)

# refine (task-specific)
task_boxes(slide1, y, ["det_refine", "map_refine", "plan_refine", "ego_refine"],
           C_TASK, C_TASK_BORDER, font_size=8)
y += BOX_H + ROW_GAP

# ── Legend (bottom right of slide 1) ──
leg_x = Inches(10.5)
leg_y = Inches(1.2)
leg_items = [
    (C_GRP_A, C_GRP_A_BORDER, "Group A: backbone+neck"),
    (C_GRP_B, C_GRP_B_BORDER, "Group B: norm+ffn (CORE)"),
    (C_GRP_C, C_GRP_C_BORDER, "Group C: fc_before/after"),
    (C_GRP_D, C_GRP_D_BORDER, "Group D: inter_gnn"),
    (C_TASK, C_TASK_BORDER, "Task-specific (not measured)"),
    (C_NOT_MEASURED, LIGHT_GRAY, "Separate attn (not measured)"),
]
add_text(slide1, leg_x, leg_y - Inches(0.3), Inches(2.5), Inches(0.25),
         "Legend", font_size=10, bold=True)
for i, (fill, border, label) in enumerate(leg_items):
    ly = leg_y + i * Inches(0.32)
    add_box(slide1, leg_x, ly, Inches(0.3), Inches(0.22), "", fill, border, font_size=6)
    add_text(slide1, leg_x + Inches(0.35), ly, Inches(2.2), Inches(0.22),
             label, font_size=8)


# ════════════════════════════════════════════════════════════════
# Helper: Group detail slide
# ════════════════════════════════════════════════════════════════
def make_group_slide(group_letter, group_name, color, border_color, layers_info,
                     sharing_level, why_measure, what_flows, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    # Title bar
    add_box(slide, Inches(0.3), Inches(0.2), Inches(12.7), Inches(0.6),
            f"Group {group_letter}: {group_name}",
            border_color, font_size=20, bold=True, font_color=WHITE)

    # ── Left column: Layer details ──
    lx = Inches(0.5)
    ly = Inches(1.1)

    add_text(slide, lx, ly, Inches(5), Inches(0.3),
             "Layers in this group:", font_size=13, bold=True)
    ly += Inches(0.4)

    for layer_name, layer_desc, layer_params in layers_info:
        add_box(slide, lx, ly, Inches(5.5), Inches(0.5),
                f"{layer_name}", color, border_color, font_size=11, bold=True)
        add_text(slide, lx + Inches(5.7), ly, Inches(4), Inches(0.5),
                 layer_desc, font_size=9, color=DARK_GRAY)
        if layer_params:
            add_text(slide, lx + Inches(5.7), ly + Inches(0.22), Inches(4), Inches(0.25),
                     layer_params, font_size=8, color=RGBColor(120, 120, 120))
        ly += Inches(0.6)

    # ── Sharing level ──
    ly += Inches(0.2)
    add_text(slide, lx, ly, Inches(2), Inches(0.3),
             "Sharing Level:", font_size=12, bold=True)
    add_text(slide, lx + Inches(2), ly, Inches(8), Inches(0.3),
             sharing_level, font_size=11, color=border_color, bold=True)
    ly += Inches(0.45)

    # ── Why measure ──
    add_text(slide, lx, ly, Inches(2), Inches(0.3),
             "Why measure:", font_size=12, bold=True)
    ly += Inches(0.35)
    for line in why_measure:
        add_text(slide, lx + Inches(0.3), ly, Inches(10), Inches(0.25),
                 f"\u2022 {line}", font_size=10)
        ly += Inches(0.3)

    # ── Data flow ──
    ly += Inches(0.15)
    add_text(slide, lx, ly, Inches(3), Inches(0.3),
             "Data Flow:", font_size=12, bold=True)
    ly += Inches(0.35)
    for line in what_flows:
        add_text(slide, lx + Inches(0.3), ly, Inches(11), Inches(0.25),
                 line, font_size=10, color=DARK_GRAY)
        ly += Inches(0.3)

    # ── Notes box ──
    ly += Inches(0.2)
    note_h = Inches(0.3) * len(notes) + Inches(0.3)
    add_box(slide, lx, ly, Inches(11.5), note_h,
            "", RGBColor(255, 253, 231), RGBColor(200, 180, 0), font_size=9,
            align=PP_ALIGN.LEFT)
    ny = ly + Inches(0.1)
    add_text(slide, lx + Inches(0.15), ny, Inches(1), Inches(0.2),
             "Notes:", font_size=9, bold=True, color=RGBColor(150, 130, 0))
    ny += Inches(0.25)
    for note in notes:
        add_text(slide, lx + Inches(0.3), ny, Inches(11), Inches(0.22),
                 f"\u2022 {note}", font_size=9, color=RGBColor(80, 80, 80))
        ny += Inches(0.25)

    return slide


# ════════════════════════════════════════════════════════════════
# Slide 2: Group A - Backbone + Neck
# ════════════════════════════════════════════════════════════════
make_group_slide(
    "A", "Backbone + Neck", C_GRP_A, C_GRP_A_BORDER,
    layers_info=[
        ("img_backbone (ResNet-50)",
         "Pretrained image feature extractor",
         "~23.5M params, 4 residual stages"),
        ("img_neck (FPN)",
         "Feature Pyramid Network for multi-scale features",
         "~6.5M params, 4 lateral + 4 fpn convs"),
    ],
    sharing_level="ALL tasks share (complete sharing)",
    why_measure=[
        "All task losses backprop through these layers",
        "If task gradients conflict here, image features get corrupted",
        "Stage1 pretrained weights may be degraded by new task gradients",
    ],
    what_flows=[
        "Camera Images \u2192 ResNet-50 \u2192 FPN \u2192 Multi-scale Features",
        "                                   \u2514\u2192 used by ALL task deformable modules",
    ],
    notes=[
        "Task head\uc5d0\uc11c \uac00\uc7a5 \uba3c \uc704\uce58 \u2192 gradient\uac00 \ud76c\uc11d(dilute)\ub420 \uc218 \uc788\uc74c",
        "Stage1\uc5d0\uc11c \ucda9\ubd84\ud788 \ud559\uc2b5\ub41c weight \u2192 stage2\uc5d0\uc11c \ud070 \ubcc0\ud654\uac00 \uc624\uba74 \ubb38\uc81c",
        "depth_loss\ub3c4 backbone\uc73c\ub85c \uc9c1\uc811 grad \uc804\ud30c (auxiliary supervision)",
    ],
)

# ════════════════════════════════════════════════════════════════
# Slide 3: Group B - Decoder Norm + FFN
# ════════════════════════════════════════════════════════════════
make_group_slide(
    "B", "Decoder Norm + FFN (Core Shared Layers)", C_GRP_B, C_GRP_B_BORDER,
    layers_info=[
        ("norm (LayerNorm)",
         "All task queries concatenated \u2192 same LN weights applied",
         "12 instances across 6 cycles, indices: [3,8,15,20,27,32,39,44,51,56,63,68]"),
        ("ffn (AsymmetricFFN)",
         "All task queries concatenated \u2192 same FFN weights applied",
         "6 instances across 6 cycles, indices: [7,19,31,43,55,67]"),
    ],
    sharing_level="ALL tasks share (complete sharing) \u2014 CORE CONFLICT ZONE",
    why_measure=[
        "ALL task queries are concatenated [det|map|plan|ego] and processed by identical weights",
        "If det gradient pulls norm/ffn in direction A, but plan pulls in direction -A \u2192 direct conflict",
        "This is the most likely cause of stage1 performance degradation in stage2",
        "FFN has large parameter count \u2192 high capacity for conflict",
    ],
    what_flows=[
        "CONCAT [det|map|plan|ego] \u2192 norm \u2192 (split\u2192deformable\u2192concat) \u2192 ffn \u2192 norm \u2192 SPLIT",
        "det query\uc640 plan query\uac00 \uac19\uc740 FFN weight\ub97c \ud1b5\uacfc \u2192 gradient \ubc29\ud5a5 \ucda9\ub3cc \uc2dc \uc591\ucabd \ubaa8\ub450 \uc131\ub2a5 \uc800\ud558",
    ],
    notes=[
        "Section 2 \uacb0\uacfc: map vs plan cos_sim = -0.0998 (\uac00\uc7a5 \ud070 conflict) \u2192 \uc774 \ub808\uc774\uc5b4\uc5d0\uc11c \ud655\uc778 \ud544\uc694",
        "Epoch \uc9c4\ud589\uc5d0 \ub530\ub77c conflict\uac00 \uc99d\uac00/\uac10\uc18c\ud558\ub294\uc9c0 \ucd94\uc801 (1ep, 2ep, 3ep)",
        "det vs plan, det vs map conflict\ub3c4 \uc5ec\uae30\uc11c \ubc1c\uc0dd\ud560 \uac00\ub2a5\uc131 \ub192\uc74c",
    ],
)

# ════════════════════════════════════════════════════════════════
# Slide 4: Group C - fc_before + fc_after
# ════════════════════════════════════════════════════════════════
make_group_slide(
    "C", "fc_before + fc_after (Decouple Attention Projection)", C_GRP_C, C_GRP_C_BORDER,
    layers_info=[
        ("fc_before  Linear(256\u2192512)",
         "Value feature dimension doubling before attention",
         "256\u00d7512 = 131,072 params (no bias)"),
        ("fc_after  Linear(512\u2192256)",
         "Attention output projection back to embed_dims",
         "512\u00d7256 = 131,072 params (no bias)"),
    ],
    sharing_level="ALL tasks share (complete sharing)",
    why_measure=[
        "decouple_attn=True \u2192 attention value/output\uc774 \uc774 projection\uc744 \uacf5\uc720",
        "gnn/inter_gnn \ubaa8\ub450 \uc774 fc\ub97c \uacbd\uc720 \u2192 \ubaa8\ub4e0 attention\uc758 bottleneck",
        "Parameter \uc218\ub294 \uc801\uc9c0\ub9cc \ubaa8\ub4e0 attention module\uc774 \uacf5\uc720\ud558\ubbc0\ub85c \uc601\ud5a5\ub825 \ud07c",
    ],
    what_flows=[
        "query \u2192 fc_before(256\u2192512) \u2192 [gnn / inter_gnn attention] \u2192 fc_after(512\u2192256) \u2192 output",
        "\ubaa8\ub4e0 task\uc758 attention value\uac00 \ub3d9\uc77c\ud55c projection\uc744 \uacf5\uc720",
    ],
    notes=[
        "independent_gnn=True\uc774\uc9c0\ub9cc fc_before/fc_after\ub294 \ud558\ub098\ub9cc \uc874\uc7ac",
        "gnn\uc774 per-group weight\uc5ec\ub3c4 fc_before/after\uc5d0\uc11c \uac04\uc811\uc801 \ucda9\ub3cc \uac00\ub2a5",
        "\uc791\uc740 parameter \uc218(262K) \ub300\ube44 \ub192\uc740 gradient traffic",
    ],
)

# ════════════════════════════════════════════════════════════════
# Slide 5: Group D - inter_gnn
# ════════════════════════════════════════════════════════════════
make_group_slide(
    "D", "inter_gnn (Interactive Cross-Task Attention)", C_GRP_D, C_GRP_D_BORDER,
    layers_info=[
        ("inter_gnn (InteractiveAttention)",
         "plan/ego queries attend TO det/map features (unidirectional)",
         "6 instances, indices: [2,14,26,38,50,62], MultiheadFlashAttention(256-dim)"),
    ],
    sharing_level="Cross-task interaction (plan/ego \u2192 det/map, unidirectional)",
    why_measure=[
        "plan/ego\uac00 det/map feature\ub97c key/value\ub85c \uc0ac\uc6a9 \u2192 plan loss\uac00 det/map representation\uc744 \uc65c\uace1\ud560 \uc218 \uc788\uc74c",
        "plan loss backprop \u2192 inter_gnn K/V projection \u2192 det/map feature\uc5d0 gradient \uc804\ud30c",
        "det/map\uc758 \uc131\ub2a5 \uc800\ud558\uac00 \uc774 \uacbd\ub85c\ub97c \ud1b5\ud574 \ubc1c\uc0dd\ud560 \uc218 \uc788\uc74c",
    ],
    what_flows=[
        "Query: [plan, ego] queries",
        "Key/Value: [det, map] features  \u2190 plan loss\uac00 \uc5ec\uae30\ub85c backprop",
        "Output: plan/ego\uc758 updated features (det/map\ub294 \ubcc0\uacbd\ub418\uc9c0 \uc54a\uc74c, \ud558\uc9c0\ub9cc gradient\ub294 \uc804\ud30c)",
    ],
    notes=[
        "\ub2e8\ubc29\ud5a5: plan/ego\uac00 det/map\uc744 attend. det/map\uc740 plan/ego\ub97c attend\ud558\uc9c0 \uc54a\uc74c",
        "Forward\uc5d0\uc11c det/map feature\ub294 \ubd88\ubcc0\uc774\uc9c0\ub9cc, backward\uc5d0\uc11c gradient\ub294 det/map\uc73c\ub85c \uc804\ud30c",
        "temp_gnn group2 (plan/ego\u2194det,map)\ub3c4 \uc720\uc0ac\ud55c \uad6c\uc870\uc774\uc9c0\ub9cc \ubcc4\ub3c4 \uce21\uc815 \ub300\uc0c1\uc5d0\uc11c\ub294 \uc81c\uc678",
        "motion\uc740 query_select\uc5d0 \uc5c6\uc73c\ubbc0\ub85c inter_gnn\uc744 \uacbd\uc720\ud558\uc9c0 \uc54a\uc74c (det\uc5d0\uc11c \ud30c\uc0dd)",
    ],
)

# ── Save ──
out_path = os.path.join(
    '/home/kyungmin/min_ws/rideflux/HiP-AD/notebooks',
    'hipad_architecture_gradient_conflict.pptx'
)
prs.save(out_path)
print(f"Saved: {out_path}")
