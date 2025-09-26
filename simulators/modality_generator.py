import numpy as np
from scipy.stats import multivariate_normal

class ModalityGenerator:
    def __init__(self, dims):
        self.dims = dims
        self.W, self.H = dims
        self.x, self.y = np.mgrid[0:self.W, 0:self.H]
        self.pos = np.dstack((self.x, self.y))

    def _poisson_disk_sample(self, min_dist, num_sources):
        points = []
        for _ in range(num_sources * 100): # 增加尝试次数以确保成功
            if len(points) >= num_sources: break
            candidate = [np.random.uniform(0, self.W), np.random.uniform(0, self.H)]
            is_valid = True
            if not points:
                points.append(candidate)
                continue
            for point in points:
                if np.linalg.norm(np.array(candidate) - np.array(point)) < min_dist:
                    is_valid = False
                    break
            if is_valid:
                points.append(candidate)
        return points

    def _create_gaussian_field(self, sources):
        field = np.zeros(self.dims)
        if not sources: return field
        for src in sources:
            mean = src['center']
            cov = [[src['scale_x']**2, 0], [0, src['scale_y']**2]]
            rv = multivariate_normal(mean, cov)
            field += rv.pdf(self.pos) * src['amplitude']
        if np.max(field) > 0:
            field /= np.max(field)
        return field

    def generate_modalities(self, params):
        modalities = {}
        modalities['static_obstacles'] = self._create_gaussian_field(params.get('obstacles', []))
        modalities['no_fly_zones'] = self._create_gaussian_field(params.get('nfz', []))
        modalities['radar_coverage'] = self._create_gaussian_field(params.get('radars', []))
        modalities['camera_coverage'] = self._create_gaussian_field(params.get('cameras', []))
        modalities['hostile_threat'] = self._create_gaussian_field(params.get('hostiles', []))
        
        comm_fields = []
        for station in params.get('comm_stations', []):
            dist = np.linalg.norm(self.pos - station['center'], axis=2)
            field = 1.0 - np.exp(-dist / station['radius'])
            comm_fields.append(field)
        
        if comm_fields:
            modalities['comm_signal'] = np.min(comm_fields, axis=0)
        else:
            modalities['comm_signal'] = np.ones(self.dims)

        modalities['uav_state'] = {'battery': params.get('battery_level', 1.0)}
        return modalities
