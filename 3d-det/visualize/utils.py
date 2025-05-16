import math
import matplotlib.pyplot as plt
import numpy as np
import yaml
import lanelet2
import numpy as np
from copy import deepcopy

class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.o = o      # orientation
        self.s = None   # detection score
    
    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.o}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.h = data['height']
        bbox.w = data['width']
        bbox.l = data['length']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox
    
    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return
    
    @classmethod
    def box2corners2d(cls, bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1     
    
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    
    @classmethod
    def box2corners3d(cls, bbox):
        """ the coordinates for bottom corners
        """
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()
    
    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result
    
    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result
    
    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox 
    
    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs,
                                  np.ones(pcs.shape[0])[:, np.newaxis]),
                                  axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs
    
    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw
    
    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
        
        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result


def scalar_to_color(scalar_value,scalar_min=0,scalar_max=100):
    cmap = plt.cm.get_cmap('gist_rainbow')  # Choose the Gist color map (you can change it to 'gist_ncar' or any other available colormap)

    # Example usage:
    #scalar_min = 0  # Minimum value of the scalar range
    #scalar_max = 100  # Maximum value of the scalar range

    normalized_value = (scalar_value - scalar_min) / (scalar_max - scalar_min)  # Normalize the scalar value between 0 and 1
    rgba_color = cmap(normalized_value)  # Obtain the RGBA color corresponding to the normalized scalar value

    return rgba_color


def latlon2utm(lat, lon):
    # Constants
    a = 6378137.0  # WGS84 major axis
    f = 1 / 298.257223563  # WGS84 flattening
    k0 = 0.9996  # scale factor

    # Calculate the UTM Zone
    zone_number = int((lon + 180) / 6) + 1

    # Calculate central meridian for the zone
    lambda0 = math.radians((zone_number - 1) * 6 - 180 + 3)

    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    e = math.sqrt(1 - (1 - f) * (1 - f))
    N = a / math.sqrt(1 - e * e * math.sin(lat_rad) * math.sin(lat_rad))
    T = math.tan(lat_rad) * math.tan(lat_rad)
    C = (e * e / (1 - e * e)) * math.cos(lat_rad) * math.cos(lat_rad)
    A = (lon_rad - lambda0) * math.cos(lat_rad)

    M = a * ((1 - e * e / 4 - 3 * e * e * e * e / 64 - 5 * e * e * e * e * e * e / 256) * lat_rad
            - (3 * e * e / 8 + 3 * e * e * e * e / 32 + 45 * e * e * e * e * e * e / 1024) * math.sin(2 * lat_rad)
            + (15 * e * e * e * e / 256 + 45 * e * e * e * e * e * e / 1024) * math.sin(4 * lat_rad)
            - (35 * e * e * e * e * e * e / 3072) * math.sin(6 * lat_rad))

    x = k0 * N * (A + (1 - T + C) * A * A * A / 6
                + (5 - 18 * T + T * T + 72 * C - 58 * e * e / (1 - e * e)) * A * A * A * A * A / 120) + 500000.0

    y = k0 * (M + N * math.tan(lat_rad) * (A * A / 2
                + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24
                + (61 - 58 * T + T * T + 600 * C - 330 * e * e / (1 - e * e)) * A * A * A * A * A * A / 720))

    if lat < 0:
        y += 10000000.0  # 10000000 meter offset for southern hemisphere

    return zone_number, x, y


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8)):
        self.figure = plt.figure(name, figsize=figsize)
        plt.axis('equal')
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'blue': np.array([0, 0, 256]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([0, 256, 0]) / 256
        }

        self.utm_to_l2 = np.load('/net/ca-home1/home/ma/yajmera/VRUBIS/metadata/maptolidar2.npy')
        fp = '/net/ca-home1/home/ma/yajmera/VRUBIS/metadata/lanelet2_map'
        osm_path = '{}/lanelet2_map.osm'.format(fp)
        config_path = '{}/map_config.yaml'.format(fp)

        with open(config_path) as stream:
            config = yaml.full_load(stream)

        origin = np.array([config['/**']['ros__parameters']['map_origin']['latitude'],config['/**']['ros__parameters']['map_origin']['longitude']])
        self.origin_utm = np.array(latlon2utm(origin[0],origin[1])[1:])
        self.ll2_api = lanelet2.io.load(osm_path,lanelet2.projection.UtmProjector(lanelet2.io.Origin(origin[0], origin[1])))   
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, intensity, color='black'):   
        # Plots the point cloud     
        vis_pc = np.asarray(pc)
        color = scalar_to_color(intensity,scalar_min=0,scalar_max=255)                
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=color[:,:3], s=0.01)

    def handler_map(self):
        # Plots the lanelet2 map 
        for laneii, lane in enumerate(self.ll2_api.laneletLayer):
            left_bound_utm = np.array([np.array([point.x, point.y])+self.origin_utm for point in self.ll2_api.laneletLayer[lane.id].leftBound])
            right_bound_utm = np.array([np.array([point.x, point.y])+self.origin_utm for point in self.ll2_api.laneletLayer[lane.id].rightBound])
            centerline_utm = np.array([np.array([point.x, point.y])+self.origin_utm for point in self.ll2_api.laneletLayer[lane.id].centerline])

            left_bound_utm = np.hstack((left_bound_utm, np.zeros_like(left_bound_utm[:,0:1]), np.ones_like(left_bound_utm[:,0:1])))
            right_bound_utm = np.hstack((right_bound_utm, np.zeros_like(right_bound_utm[:,0:1]), np.ones_like(right_bound_utm[:,0:1])))
            centerline_utm = np.hstack((centerline_utm, np.zeros_like(centerline_utm[:,0:1]), np.ones_like(centerline_utm[:,0:1])))

            left_bound_l2 = self.utm_to_l2.dot(left_bound_utm.T).T
            right_bound_l2 = self.utm_to_l2.dot(right_bound_utm.T).T            
            polygon_l2 = np.vstack([left_bound_l2,right_bound_l2[::-1]])

            if lane.attributes['subtype'] == 'road':
                plt.plot(left_bound_l2[:,0], left_bound_l2[:,1], color=(0,0,0), zorder=1)
                plt.plot(right_bound_l2[:,0], right_bound_l2[:,1], color=(0,0,0), zorder=1)
                #plt.plot(centerline_l2[:,0], centerline_l2[:,1], color=(0.6,0.6,0.6), zorder=1)
                plt.fill(polygon_l2[:,0], polygon_l2[:,1], color=(0,0,0), alpha=0.05, zorder=1)
            else:
                plt.plot(left_bound_l2[:,0], left_bound_l2[:,1], color=(0,0,1), zorder=2)
                plt.plot(right_bound_l2[:,0], right_bound_l2[:,1], color=(0,0,1), zorder=2)
                plt.fill(polygon_l2[:,0], polygon_l2[:,1], color=(0,0,1), alpha=0.05, zorder=2)
        
        for way in self.ll2_api.lineStringLayer:
            if self.ll2_api.lineStringLayer[way.id].attributes['type'] == 'curbstone':
                way_utm = np.array([np.array([point.x, point.y])+self.origin_utm for point in self.ll2_api.lineStringLayer[way.id]])
                way_utm = np.hstack((way_utm, np.zeros_like(way_utm[:,0:1]), np.ones_like(way_utm[:,0:1])))
                way_l2 = self.utm_to_l2.dot(way_utm.T).T
                plt.plot(way_l2[:,0], way_l2[:,1], color=(1,0,0), zorder=2)
                plt.fill(way_l2[:,0], way_l2[:,1], color=(1,0,0), alpha=0.05, zorder=2)

            
    def handler_box(self, box: BBox, message: str='', color='red', linestyle='solid',addtext=True):
        # Plots the bounding box
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        corner_index = 0
        if addtext:
            plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP['blue'],fontsize=18)
    
    def hanlder_setlimit(self,xmin,ymin,xmax,ymax):
        ax = plt.gca()
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])