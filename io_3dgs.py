import numpy as np
from plyfile import PlyData, PlyElement

"""
Tools for handling Gaussian models stored in PLY files
- load_gaussian_ply()
- export_gs_to_ply()
- extract_gaussians()
- copy_gs()
- add_attribute()
- delete_attribute()
- concatenate_gs()
- get_AABB()
"""

class GaussianModelV2():
    """
    A class to load and manipulate the more flexible Gaussian models from PLY files.
    """
    data = {}
    num_of_point = 0
    sh_deg = -1

    def __init__(self, gs_path: str):
        self.load_gaussian_ply(gs_path)

    def copy_gs(self):
        new_gs = GaussianModelV2.__new__(GaussianModelV2)
        new_gs.data = self.data.copy()
        new_gs.num_of_point = self.num_of_point
        new_gs.sh_deg = self.sh_deg
        return new_gs    

    def load_gaussian_ply(self, path: str) -> None:
        self.data = {}
        plydata = PlyData.read(path)
        self.num_of_point = plydata.elements[0][plydata.elements[0].properties[0].name].shape[0]
        for property in plydata.elements[0].properties:
            _property_data = {"val_dtype": property.val_dtype, "data": np.asarray(plydata.elements[0][property.name])}
            self.data.update({property.name: _property_data})
        data_keys = self.data.keys()
        if all(item in data_keys for item in ['f_dc_0', 'f_dc_1', 'f_dc_2']):
            self.sh_deg = 0
            if all(item in data_keys for item in [f'f_rest_{i}' for i in range(0*3, 3*3)]):
                self.sh_deg = 1
                if all(item in data_keys for item in [f'f_rest_{i}' for i in range(3*3, 3*3+5*3)]):
                    self.sh_deg = 2
                    if all(item in data_keys for item in [f'f_rest_{i}' for i in range(3*3+5*3, (3*3+5*3)+7*3)]):
                        self.sh_deg = 3
    
    def add_attribute(self, key: str, type: str, data: np.ndarray):
        if self.num_of_point == data.shape[0]:
            _property_data = {"val_dtype": type, "data": data}
            self.data.update({key: _property_data})
        else:
            print("Can't add attribute. Wrong number of points.")

    def delete_attribute(self, key: str):
        if key in self.data:
            del self.data[key]
        else:
            print(f"Attribute {key} not found in the model.")

    def extract_gaussians(self, indices: list):
        """
        Extracts a subset of Gaussians based on the provided indices.
        Returns a new GaussianModelV2 instance with the extracted data.
        """
        new_gs = GaussianModelV2.__new__(GaussianModelV2)
        new_gs.data = {key: {"val_dtype": value["val_dtype"], "data": value["data"][indices]} for key, value in self.data.items()}
        new_gs.num_of_point = len(indices)
        new_gs.sh_deg = self.sh_deg
        return new_gs

    def export_gs_to_ply(self, save_path: str, ascii=False):
        
        assert hasattr(self, 'data'), "self.data must be loaded via load_gaussian_ply()"
        assert hasattr(self, 'num_of_point'), "self.num_of_point not found"
        
        data_full = []
        dtype_full = []
        
        for key, value in self.data.items():
            dtype_full.append((key, value["val_dtype"]))
            array = value["data"]
            if array.ndim == 1:
                array = array.reshape(-1, 1)  # 轉成 (N,1)
            data_full.append(array)
        
        attributes = np.concatenate(data_full, axis=1)
        elements = np.empty(self.num_of_point, dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        if ascii:
            PlyData([el], text=True).write(save_path)
        else:
            PlyData([el]).write(save_path)

    def concatenate_gs(self, other_gs):
        """
        Concatenates another GaussianModelV2 instance to the current one.
        
        [POS_s][COLOR_o]...[ROT_o]
                   +
        [POS_o][COLOR_o]...[ROT_o]
        """
        
        if not isinstance(other_gs, GaussianModelV2):
            raise TypeError("other_gs must be an instance of GaussianModelV2")
        
        new_num_of_points = self.num_of_point + other_gs.num_of_point

        # Check if keys match
        if set(self.data.keys()) != set(other_gs.data.keys()):
            raise ValueError("The two Gaussian models must have the same attributes to concatenate.")
        
        for key in self.data.keys():
            self.data[key]["data"] = np.concatenate((self.data[key]["data"], other_gs.data[key]["data"]), axis=0)
            if self.data[key]["data"].shape[0] != new_num_of_points:
                raise ValueError(f"Concatenation failed for attribute {key}. Expected {new_num_of_points} points, got {self.data[key]['data'].shape[0]}")
        
        self.num_of_point = new_num_of_points
        
    def get_AABB(self):
        """
        Compute the Axis-Aligned Bounding Box (AABB) of the Gaussian model.
        Returns min_bound and max_bound as numpy arrays.
        """
        if 'x' not in self.data or 'y' not in self.data or 'z' not in self.data:
            raise ValueError("Position attributes 'x', 'y', 'z' are required to compute AABB.")
        
        x = self.data['x']['data']
        y = self.data['y']['data']
        z = self.data['z']['data']
        
        min_bound = np.array([np.min(x), np.min(y), np.min(z)])
        max_bound = np.array([np.max(x), np.max(y), np.max(z)])
        
        return min_bound, max_bound