import numpy as np

def generate_heuristic_labels(modalities, risk_types, weights, thresholds):
    labels = {}
    dims = modalities.get('comm_signal', np.ones((100,100))).shape

    static_obstacles = modalities.get('static_obstacles', np.zeros(dims))
    no_fly_zones = modalities.get('no_fly_zones', np.zeros(dims))
    labels['collision'] = np.maximum(static_obstacles, no_fly_zones)

    radar_coverage = modalities.get('radar_coverage', np.zeros(dims))
    camera_coverage = modalities.get('camera_coverage', np.zeros(dims))
    labels['exposure'] = np.maximum(radar_coverage, camera_coverage)

    battery_level = modalities['uav_state']['battery']
    labels['energy'] = np.full(dims, (1.0 - battery_level) ** 2)

    labels['communication'] = modalities.get('comm_signal', np.ones(dims))
    labels['threat'] = modalities.get('hostile_threat', np.zeros(dims))
    
    # 将所有标签堆叠成一个 [W, H, num_risks] 的张量
    stacked_labels = np.stack([labels[risk] for risk in risk_types], axis=-1)
    
    # 归一化每个风险通道
    for i in range(stacked_labels.shape[-1]):
        max_val = np.max(stacked_labels[..., i])
        if max_val > 0:
            stacked_labels[..., i] /= max_val
            
    return stacked_labels
