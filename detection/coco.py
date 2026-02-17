from typing import Dict, List, Any
import yaml


class CocoClassMapper:
    def __init__(self, coco_yaml_path: str):
        self.class_id_to_name = self._load_class_names(coco_yaml_path)

    @staticmethod
    def _load_class_names(yaml_path: str) -> Dict[int, str]:
        """
        Loads the class ID to name mapping from the coco.yaml file.

        :param yaml_path: Path to the coco.yaml file.
        :return: A dictionary mapping class IDs to their names.
        """
        try:
            with open(yaml_path, "r") as file:
                data = yaml.safe_load(file)
                return {int(k): v for k, v in data["names"].items()}
        except Exception:
            raise

    def map_class_names(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Maps class IDs in the detections to their corresponding class names.

        :param detections: A list of detection dictionaries.
        :return: The enriched list of detections with class names.
        """
        for detection in detections:
            class_id = detection.get("class_id") - 1
            detection["class_name"] = self.class_id_to_name.get(class_id, "Unknown")
        return detections
