import numpy as np
import os
import json
from tqdm import tqdm

from simulators.modality_generator import ModalityGenerator
from core.heuristic_labels import generate_heuristic_labels

with open('configs/default_config.json', 'r') as f:
    config = json.load(f)
with open('configs/scenario_weights.json', 'r') as f:
    scenario_configs = json.load(f)

DIMS, RISK_TYPES = config['grid']['dimensions'], config['risk_types']
W, H = DIMS
NUM_SAMPLES_PER_SCENARIO = 3000
OUTPUT_DIR = "training_data_xai"

mod_gen = ModalityGenerator(DIMS)
print(f"开始为所有场景生成结构化训练样本...")

for scenario_name, scenario_config in scenario_configs.items():
    scenario_output_dir = os.path.join(OUTPUT_DIR, scenario_name)
    if not os.path.exists(scenario_output_dir): os.makedirs(scenario_output_dir)
    
    print(f"\n--- 正在生成 '{scenario_name}' 场景数据 ---")
    for i in tqdm(range(NUM_SAMPLES_PER_SCENARIO), desc=f"生成 {scenario_name} 数据"):
        sim_params = {}
        if scenario_name == 'urban':
            num_obstacles = np.random.randint(3, 6)
            sim_params['obstacles'] = [{'center': p, 'scale_x': np.random.uniform(10, 15), 'scale_y': np.random.uniform(10, 15), 'amplitude': np.random.uniform(0.8, 1.0)} for p in mod_gen._poisson_disk_sample(W/5, num_obstacles)]
            num_cameras = np.random.randint(2, 4)
            sim_params['cameras'] = [{'center': p, 'scale_x': 15, 'scale_y': 15, 'amplitude': 1} for p in mod_gen._poisson_disk_sample(W/4, num_cameras)]
            num_comms = np.random.randint(3, 5)
            sim_params['comm_stations'] = [{'center': p, 'radius': W/2} for p in mod_gen._poisson_disk_sample(W/2, num_comms)]
        elif scenario_name == 'military':
            base_center = [np.random.uniform(W/4, 3*W/4), np.random.uniform(H/4, 3*H/4)]
            sim_params['obstacles'] = [{'center': base_center, 'scale_x': 15, 'scale_y': 15, 'amplitude': 1}]
            sim_params['nfz'] = [{'center': base_center, 'scale_x': 30, 'scale_y': 30, 'amplitude': 1}]
            sim_params['radars'] = [{'center': base_center, 'scale_x': 40, 'scale_y': 40, 'amplitude': 1}]
            num_hostiles = np.random.randint(1, 3)
            sim_params['hostiles'] = [{'center': [base_center[0] + np.random.uniform(-20, 20), base_center[1] + np.random.uniform(-20, 20)], 'scale_x': 10, 'scale_y': 10, 'amplitude': 1} for _ in range(num_hostiles)]
            num_comms = np.random.randint(1, 2)
            sim_params['comm_stations'] = [{'center': p, 'radius': W/3} for p in mod_gen._poisson_disk_sample(W/1.5, num_comms)]
        elif scenario_name == 'rural':
            num_comms = np.random.randint(0, 2)
            if num_comms > 0:
                sim_params['comm_stations'] = [{'center': p, 'radius': W/2.5} for p in mod_gen._poisson_disk_sample(W/1.2, num_comms)]

        sim_params['battery_level'] = np.random.uniform(0.2, 1.0)
        modalities = mod_gen.generate_modalities(sim_params)
        labels = generate_heuristic_labels(modalities, RISK_TYPES, scenario_config['weights'], scenario_config['thresholds'])
        
        input_data_np = np.stack([
            modalities.get('static_obstacles', np.zeros(DIMS)), modalities.get('no_fly_zones', np.zeros(DIMS)),
            modalities.get('radar_coverage', np.zeros(DIMS)), modalities.get('camera_coverage', np.zeros(DIMS)),
            modalities.get('comm_signal', np.ones(DIMS)), modalities.get('hostile_threat', np.zeros(DIMS)),
            np.full(DIMS, modalities['uav_state']['battery'])
        ], axis=-1)
        np.savez_compressed(
            os.path.join(scenario_output_dir, f"sample_{i:05d}.npz"),
            data=input_data_np.astype(np.float16),
            label=labels.astype(np.float16)
        )
print("\n所有场景的数据已生成并分类保存。")
