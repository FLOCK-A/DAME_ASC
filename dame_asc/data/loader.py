from typing import Dict, List, Optional
from pathlib import Path
import json


class DataLoader:
    """DataLoader that reads JSON manifest files.

    Supports two manifest formats:
      - dict-list (default): list of dicts with keys: path, scene, device, city(optional)
      - list4: list of lists: [path, city, scene, device]

    It can also load label_map and device_map JSON files.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        manifest_format: str = "dict",
        label_map: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest_format = manifest_format
        self.label_map_path = Path(label_map) if label_map else None
        self.device_map_path = Path(device_map) if device_map else None
        self._label_map = None
        self._device_map = None

    def load_label_map(self) -> Dict[str, int]:
        if self._label_map is not None:
            return self._label_map
        if not self.label_map_path:
            self._label_map = {}
            return self._label_map
        with open(self.label_map_path, "r", encoding="utf-8") as f:
            self._label_map = json.load(f)
        return self._label_map

    def load_device_map(self) -> Dict[str, int]:
        if self._device_map is not None:
            return self._device_map
        if not self.device_map_path:
            self._device_map = {}
            return self._device_map
        with open(self.device_map_path, "r", encoding="utf-8") as f:
            self._device_map = json.load(f)
        # Ensure unknown present
        if "unknown" not in self._device_map:
            self._device_map["unknown"] = -1
        return self._device_map

    def _parse_list4_item(self, item: List) -> Dict:
        # [path, city, scene, device]
        path = item[0]
        city = int(item[1]) if len(item) > 1 else None
        scene = int(item[2]) if len(item) > 2 else None
        device = int(item[3]) if len(item) > 3 else None
        return {"path": path, "scene": scene, "device": device, "city": city}

    def _parse_dict_item(self, item: Dict) -> Dict:
        # Expect at least path and scene
        return {
            "path": item.get("path"),
            "scene": item.get("scene"),
            "device": item.get("device", -1),
            "city": item.get("city"),
            "meta": item.get("meta", {}),
        }

    def load_manifest(self) -> List[Dict]:
        if not self.manifest_path or not self.manifest_path.exists():
            return []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = []
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # list-of-lists
            for it in data:
                samples.append(self._parse_list4_item(it))
        elif isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
            for it in data:
                samples.append(self._parse_dict_item(it))
        else:
            raise ValueError("Unsupported manifest format")
        return samples

    def list_samples(self) -> List[Dict]:
        return self.load_manifest()

    def generate_synthetic(self, n: int = 8) -> List[Dict]:
        samples = []
        for i in range(n):
            samples.append({"id": f"synth_{i}", "path": None, "scene": None, "device": -1, "city": None})
        return samples
