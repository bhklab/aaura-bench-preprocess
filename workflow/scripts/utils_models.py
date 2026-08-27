from typing import Optional
from pydantic import BaseModel

from pathlib import Path

class AAuraIndexRow(BaseModel):
	"""Represents a single row in the aaura index."""
	
	id: str
	image_path: Path
	mask_path: Path
	mask_idx: int = 1
	mask_voxel_label: int = 1
	size: tuple
	spacing: tuple
	origin: tuple
	direction: tuple
	mask_volume: float
	annotation_type: str = "RERECIST"
	annotation_coords: dict
	largest_slice_index: int
	centered_bbox_coords: dict
	lesion_location: Optional[str] = None
	source: str