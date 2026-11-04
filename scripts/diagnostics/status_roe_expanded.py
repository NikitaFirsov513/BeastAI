"""Compact read-only progress for the larger-data experiment."""
from project_paths import PROJECT_ROOT
from pathlib import Path
import json
ROOT=PROJECT_ROOT
for label,name,fields in [
 ('download','data/roe/expanded_candidates/status.json',('state','completed','total','range_body_MiB','error','retry_seconds')),
 ('grouping','data/roe_expanded/preparation_status.json',('state','processed','target','new_candidates_related_to_controls','unique_roe_training_images','roe_training_boxes','excluded_related_to_control','legacy_selected_now_connected_to_control','error')),
 ('dataset','data/roe_expanded/dataset_ready.json',('prepared','unique_roe_training','fresh_diagnostic_images','entries_per_epoch')),
 ('training','data/roe_expanded/training_status.json',('state','epoch_completed','epochs','elapsed_seconds','error')),
 ('comparison','data/roe_expanded/comparison_status.json',('state','stage','error'))]:
    path=ROOT/name
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'));print(label+': '+json.dumps({key:data[key] for key in fields if key in data},ensure_ascii=True))
