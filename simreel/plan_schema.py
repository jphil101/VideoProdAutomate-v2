"""The *story plan*: the only thing the planner (LLM or heuristic) has to produce.

Design rule learned from the previous pipeline: the language model never does 3D
arithmetic.  It speaks the language a site engineer uses -- gridlines, levels,
sections, sequence -- and deterministic code (structure.py / choreo.py) turns
that into exact coordinates, physics bodies, crane motions and camera moves.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ElementKind = Literal["footing", "column", "beam", "slab", "module", "wall_panel", "brace", "box"]
MaterialName = Literal["steel", "galvanized_steel", "concrete", "precast_concrete", "timber", "glass", "module"]
ShotName = Literal["establishing", "wide", "medium", "close", "top", "follow", "orbit"]
ActionName = Literal[
    "lift_and_place",  # tower crane picks the element from the laydown yard and sets it at its target
    "lower_into_place",  # element descends vertically from above (no crane visible)
    "reveal",  # element appears in place (e.g. footings, things that already exist)
    "bolt",  # connection is made: the element is fixed to what supports it
    "highlight",  # glow/pulse to draw attention, no motion
    "camera",  # camera-only beat (use `shot`)
    "stock_clip",  # cut to stock footage for this sentence (use `stock_query`)
    "pause",  # nothing new happens; hold the current view
]


class Site(BaseModel):
    environment: Literal["clear_day", "overcast", "golden_hour"] = Field(
        "clear_day", description="Lighting/sky preset")
    ground: Literal["gravel", "soil", "concrete"] = "gravel"
    crane: Literal["tower", "none"] = Field(
        "tower", description="'tower' shows a tower crane doing all lifts; 'none' lowers elements from above")
    show_grid_markings: bool = Field(True, description="Paint gridlines and grid bubbles on the ground slab")


class Grid(BaseModel):
    x_spacings_m: list[float] = Field(
        description="Distances between consecutive lettered gridlines along X. [6, 6] -> gridlines A, B, C")
    y_spacings_m: list[float] = Field(
        description="Distances between consecutive numbered gridlines along Y. [7.5] -> gridlines 1, 2")
    level_heights_m: list[float] = Field(
        description="Storey heights. Level 0 is the ground slab; level k sits at the sum of the first k heights")


class Element(BaseModel):
    id: str = Field(description="Unique, human-readable id, e.g. 'B1' or 'C-A1'")
    kind: ElementKind
    at: Optional[str] = Field(None, description="Grid node for footing/column/box, e.g. 'A1'")
    start: Optional[str] = Field(None, description="Start grid node for beam/brace/wall_panel, e.g. 'A1'")
    end: Optional[str] = Field(None, description="End grid node for beam/brace/wall_panel, e.g. 'B1'")
    bay_from: Optional[str] = Field(None, description="One corner node of the bay for slab/module, e.g. 'A1'")
    bay_to: Optional[str] = Field(None, description="Opposite corner node of the bay for slab/module, e.g. 'B2'")
    level: int = Field(0, description=(
        "column/footing/module/wall_panel/box: level its base stands on (0 = ground). "
        "beam/slab: the floor level it forms (1 = first floor). brace: level of its lower end"))
    section: Optional[str] = Field(None, description="Steel or concrete section, e.g. 'UB 406x178x60', 'SHS 300x300x12', 'W310x60', '400x400'")
    size_m: Optional[list[float]] = Field(None, description="Explicit [length_x, width_y, height_z] for box/module overrides")
    position_m: Optional[list[float]] = Field(None, description="Explicit [x, y] in metres from gridline A/1 intersection (overrides `at`)")
    offset_m: Optional[list[float]] = Field(None, description="Fine [dx, dy] shift in metres from the grid position")
    yaw_deg: Optional[float] = Field(None, description="Rotation about vertical for box elements")
    material: Optional[MaterialName] = None
    color_hex: Optional[str] = Field(None, description="Paint colour override, e.g. '#B03A2E'")
    label: Optional[str] = Field(None, description="Short on-screen callout shown when the element is installed")
    preplaced: bool = Field(False, description="True if the element already stands in place when the video starts")


class Event(BaseModel):
    sentence: int = Field(description="0-based index of the narration sentence this event illustrates")
    action: ActionName
    targets: list[str] = Field(default_factory=list, description="Element ids acted on, in installation order")
    cue_word: Optional[str] = Field(None, description=(
        "A word from that sentence; the key moment (element lands / glows) is synced to when it is spoken"))
    shot: Optional[ShotName] = None
    caption_title: Optional[str] = Field(None, description="Optional lower-third title shown during the event")
    stock_query: Optional[str] = Field(None, description="Search phrase for stock_clip events")


class StoryPlan(BaseModel):
    title: str
    site: Site = Site()
    grid: Grid
    elements: list[Element]
    events: list[Event]
