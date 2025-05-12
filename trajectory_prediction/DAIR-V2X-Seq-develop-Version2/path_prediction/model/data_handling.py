from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import math
import json

import torch
import torch.nn.functional as F

from .utils import side_to_directed_lineseg, safe_list_index

import pdb

class Map():

    def __init__(self, fp):

        #######################################################################
        # Load Map
        #######################################################################

        with open(fp, 'r') as file:
            data = json.load(file)

        self.lanelets = data.get('LANE', {})
        self.crosswalks = data.get('CROSSWALK', {})

        self.polygon_type = []
        self.polygon_interp = []
        self.crosswalk_interp = []
        self.centerline_interp = []
        self.polygon_keys = []
        self.centerline_keys = []
        self.crosswalks_keys = []

        for crosswalk_id, crosswalk in self.crosswalks.items():
            polygon = np.array([np.array(seg[1:-1].split(', '), dtype=np.float32) for seg in crosswalk['polygon']])
            self.polygon_interp.append(self.interpolate(polygon, num_pts=100)[None])
            self.polygon_keys.append(crosswalk_id)
            self.polygon_type.append('crosswalk')
            self.crosswalk_interp.append(self.interpolate(polygon, num_pts=100)[None])
            self.crosswalks_keys.append(crosswalk_id)

        for lane_segment_id, lane_segment in self.lanelets.items():
            centerline = np.array([np.array(seg[1:-1].split(', '), dtype=np.float32) for seg in lane_segment['centerline']])
            
            heading = np.arctan(centerline[1:, 1] - centerline[:-1, 1], centerline[1:, 0] - centerline[:-1, 0])
            heading = np.hstack([heading, [heading[-1]]])
            
            left_boundary = self.parallel_discrete_path(centerline[:,0], centerline[:,1], heading, -3)
            right_boundary = self.parallel_discrete_path(centerline[:,0], centerline[:,1], heading, 3)

            # self.polygon_interp.append(np.vstack((self.interpolate(left_boundary, num_pts=100),self.interpolate(right_boundary, num_pts=100)))[None])
            self.polygon_interp.append(self.interpolate(centerline, num_pts=100)[None])
            self.polygon_keys.append(lane_segment_id)
            self.polygon_type.append('centerline')
            
            self.centerline_interp.append(self.interpolate(centerline, num_pts=100)[None])
            self.centerline_keys.append(lane_segment_id)

        lane_segment_ids = list(self.lanelets.keys())
        cross_walk_ids = list(self.crosswalks.keys())

        self.polygon_interp = np.vstack(self.polygon_interp)
        self.centerline_interp = np.vstack(self.centerline_interp)
        self.crosswalk_interp = np.vstack(self.crosswalk_interp)

        #######################################################################
        # QCNet parameters
        #######################################################################

        self.vector_repr = True
        self.dim = 3
        self.num_historical_steps = 50
        self.num_future_steps = 60
        num_steps = self.num_historical_steps + self.num_future_steps

        self._agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background', 'construction', 'riderless_bicycle', 'unknown']
        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

        self.ll2_to_av2_point_types = {'unknown': 'UNKNOWN', 'virtual': 'NONE', 'stop_line': 'SOLID_WHITE', 'pedestrian_marking':'CROSSWALK', 'curbstone': 'SOLID_WHITE', 'line_thin': 'SOLID_WHITE'}
        
    def get_lanes_within(self, agent_mgrs, max_dist=100.0):
        dist = np.min(np.linalg.norm(agent_mgrs[None,None] - self.centerline_interp, axis=2),axis=1)
        mask = dist < max_dist
        idx = np.argsort(dist[mask])
        lanes = [(key,self.lanelets[key]) for i,key in enumerate(self.centerline_keys) if mask[i]]
        lanes = [lanes[i] for i in idx]
        return dict(lanes)

    def get_crosswalks_within(self, agent_mgrs, max_dist=100.0):
        dist = np.min(np.linalg.norm(agent_mgrs[None,None] - self.crosswalk_interp, axis=2),axis=1)
        mask = dist < max_dist
        idx = np.argsort(dist[mask])
        lanes = [(key,self.crosswalks[key]) for i,key in enumerate(self.crosswalks_keys) if mask[i]]
        lanes = [lanes[i] for i in idx]
        return dict(lanes)

    def get_polygons_within(self, agent_mgrs, max_dist=100.0):
        dist = np.min(np.linalg.norm(agent_mgrs[None,None] - self.polygon_interp, axis=2),axis=1)
        mask = dist < max_dist
        idx = np.argsort(dist[mask])
        lanes = [(key,(self.polygon_interp[i],self.polygon_type[i])) for i,key in enumerate(self.polygon_keys) if mask[i]]
        lanes = [lanes[i] for i in idx]
        return dict(lanes)


    def v2x_to_qcnet_polygon_type(self, v2x_type):
        if 'CITY_DRIVING' in v2x_type:
            return 'VEHICLE'
        elif 'BIKING' in v2x_type:
            return 'BIKE'
        else:
            print(f'Unknown polygon type: {v2x_type}')
            return 'VEHICLE'


    def v2x_to_qcnet_agent_types(self, v2x_type):
        v2x_type = v2x_type.lower()
        if 'vehicle' in v2x_type:
            return 'vehicle'
        elif 'bicycle' in v2x_type:
            return 'cyclist'
        elif 'pedestrian' in v2x_type:
            return 'pedestrian'
        else:
            return 'unknown'

    def get_crosswalk_edges(self, polygon):
        edges = []
        for i in range(len(polygon)):
            start_point = polygon[i]
            end_point = polygon[(i + 1) % len(polygon)]
            length = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
            edges.append({
                'start_point': start_point,
                'end_point': end_point,
                'length': length,
                'x_diff': abs(end_point[0] - start_point[0])
            })

        # Group edges by x_diff (to separate vertical edges)
        vertical_edges = sorted(edges, key=lambda e: e['x_diff'])

        # The two edges with the smallest x_diff should be the vertical edges
        left_edge, right_edge = None, None

        # Determine left and right based on the average x-coordinate
        if vertical_edges[0]['start_point'][0] < vertical_edges[1]['start_point'][0]:
            left_edge = vertical_edges[0]
            right_edge = vertical_edges[1]
        else:
            left_edge = vertical_edges[1]
            right_edge = vertical_edges[0]
        
        left_edge = np.array([left_edge['start_point'], left_edge['end_point']])
        right_edge = np.array([right_edge['start_point'], right_edge['end_point']])
        return left_edge, right_edge

    def parallel_discrete_path(self, x,y,h,offset):
        """
        Creates a parallel discrete path for a given offset.

        **Parameters:**

        - `x (float)`: The x-coordinate of the point.
        - `y (float)`: The y-coordinate of the point.
        - `h (float)`: The heading angle in radians.
        - `offset (float)`: The parallel offset distance.

        **Returns:**

        - `tuple`: A tuple containing the new x and y coordinates after applying the offset.
        """
        theta = h + np.pi / 2
        if torch.is_tensor(theta):
            x_new = x + torch.cos(theta) * offset
            y_new = y + torch.sin(theta) * offset
            return torch.hstack((x_new[:,None], y_new[:,None]))
        else:
            x_new = x + np.cos(theta) * offset
            y_new = y + np.sin(theta) * offset
            return np.hstack((x_new[:,None], y_new[:,None]))
        


    def angle_wrap(self, angle):
        """
        Wraps the input angle to the range [-π, π].

        Parameters:
        angle (float): The angle in radians to be wrapped.

        Returns:
        float: The wrapped angle in the range [-π, π].
        """
        wrapped_angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return wrapped_angle



    def interpolate(self, pts: np.ndarray, num_pts: Optional[int] = None, max_dist: Optional[float] = None) -> np.ndarray:
        """
        Interpolate points either based on cumulative distances from the first one (`num_pts`)
        or by adding extra points until neighboring points are within `max_dist` of each other.

        In particular, `num_pts` will interpolate using a variable step such that we always get
        the requested number of points.

        Args:
            pts (np.ndarray): XYZ(H) coords.
            num_pts (int, optional): Desired number of total points.
            max_dist (float, optional): Maximum distance between points of the polyline.

        Note:
            Only one of `num_pts` or `max_dist` can be specified.

        Returns:
            np.ndarray: The new interpolated coordinates.
        """
        if num_pts is not None and max_dist is not None:
            raise ValueError("Only one of num_pts or max_dist can be used!")

        if pts.ndim != 2:
            raise ValueError("pts is expected to be 2 dimensional")

        # 3 because XYZ (heading does not count as a positional distance).
        pos_dim: int = min(pts.shape[-1], 3)
        has_heading: bool = pts.shape[-1] == 4

        if num_pts is not None:
            assert num_pts > 1, f"num_pts must be at least 2, but got {num_pts}"

            if pts.shape[0] == num_pts:
                return pts

            cum_dist: np.ndarray = np.cumsum(
                np.linalg.norm(np.diff(pts[..., :pos_dim], axis=0), axis=-1)
            )
            cum_dist = np.insert(cum_dist, 0, 0)

            steps: np.ndarray = np.linspace(cum_dist[0], cum_dist[-1], num_pts)
            xyz_inter: np.ndarray = np.empty((num_pts, pts.shape[-1]), dtype=pts.dtype)
            for i in range(pos_dim):
                xyz_inter[:, i] = np.interp(steps, xp=cum_dist, fp=pts[:, i])

            if has_heading:
                # Heading, so make sure to unwrap, interpolate, and wrap it.
                xyz_inter[:, 3] = self.angle_wrap(np.interp(steps, xp=cum_dist, fp=np.unwrap(pts[:, 3])))

            return xyz_inter

        elif max_dist is not None:
            unwrapped_pts: np.ndarray = pts
            if has_heading:
                unwrapped_pts[..., 3] = np.unwrap(unwrapped_pts[..., 3])

            segments = unwrapped_pts[..., 1:, :] - unwrapped_pts[..., :-1, :]
            seg_lens = np.linalg.norm(segments[..., :pos_dim], axis=-1)
            new_pts = [unwrapped_pts[..., 0:1, :]]
            for i in range(segments.shape[-2]):
                num_extra_points = seg_lens[..., i] // max_dist
                if num_extra_points > 0:
                    step_vec = segments[..., i, :] / (num_extra_points + 1)
                    new_pts.append(
                        unwrapped_pts[..., i, np.newaxis, :]
                        + step_vec[..., np.newaxis, :]
                        * np.arange(1, num_extra_points + 1)[:, np.newaxis]
                    )

                new_pts.append(unwrapped_pts[..., i + 1 : i + 2, :])

            new_pts = np.concatenate(new_pts, axis=-2)
            if has_heading:
                new_pts[..., 3] = self.angle_wrap(new_pts[..., 3])

            return new_pts


    def get_map_features(self, pos, dist=50.0) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        crosswalks = self.get_crosswalks_within(pos, max_dist=dist)
        lanes = self.get_lanes_within(pos, max_dist=dist)

        lane_segment_ids = list(lanes.keys())
        cross_walk_ids = list(crosswalks.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment_id, lane_segment in lanes.items():

            lane_segment_idx = polygon_ids.index(lane_segment_id)
            centerline = np.array([np.array(seg[1:-1].split(', '), dtype=np.float32) for seg in lane_segment['centerline']])
            centerline = torch.from_numpy(centerline).float()
            centerline = torch.hstack([centerline, torch.zeros(centerline.size(0), 1)])
            polygon_position[lane_segment_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            polygon_type[lane_segment_idx] = self._polygon_types.index(self.v2x_to_qcnet_polygon_type(lane_segment['lane_type']))
            polygon_is_intersection[lane_segment_idx] = 1 if lane_segment['is_intersection'] else 0

            heading = torch.atan2(centerline[1:, 1] - centerline[:-1, 1], centerline[1:, 0] - centerline[:-1, 0])
            heading = torch.cat([heading, heading[-1].unsqueeze(0)])
            left_boundary = self.parallel_discrete_path(centerline[:,0], centerline[:,0], heading, -3).float()
            right_boundary = self.parallel_discrete_path(centerline[:,0], centerline[:,0], heading, 3).float()
            left_boundary = torch.hstack([left_boundary, torch.zeros(left_boundary.size(0), 1)])
            right_boundary = torch.hstack([right_boundary, torch.zeros(right_boundary.size(0), 1)])
            point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                          right_boundary[:-1, :self.dim],
                                                          centerline[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                            torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                            torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                            dim=0)
            point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                    right_vectors[:, :2],
                                                                    center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]], dim=0)
            left_type = 2 if lane_segment['lane_type'] == 'CITY_DRIVING' else 3
            right_type = 2 if lane_segment['lane_type'] == 'CITY_DRIVING' else 3
            center_type = self._point_types.index('CENTERLINE')
            point_type[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for crosswalk_id, crosswalk in crosswalks.items():

            crosswalk_idx = polygon_ids.index(crosswalk_id)
            polygon = np.array([np.array(seg[1:-1].split(', '), dtype=np.float32) for seg in crosswalk['polygon']])
            edge1, edge2 = self.get_crosswalk_edges(polygon)
            edge1 = torch.from_numpy(edge1).float()
            edge2 = torch.from_numpy(edge2).float()
            edge1 = torch.hstack([edge1, torch.zeros(edge1.size(0), 1)])
            edge2 = torch.hstack([edge2, torch.zeros(edge2.size(0), 1)])
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[:self.dim]
            polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                             (end_position - start_position)[0])
            polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                                (start_position - end_position)[0])
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
            polygon_type[crosswalk_idx] = self._polygon_types.index('PEDESTRIAN')
            polygon_type[crosswalk_idx + len(cross_walk_ids)] = self._polygon_types.index('PEDESTRIAN')
            polygon_is_intersection[crosswalk_idx] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = self._polygon_is_intersections.index(None)

            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
            centerline = self.interpolate(torch.stack([start_position, end_position], dim=0).numpy(), num_pts=max(2,num_centerline_points))
            centerline = torch.from_numpy(centerline).float()
            point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                        right_boundary[:-1, :self.dim],
                                                        centerline[:-1, :self.dim]], dim=0)
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [right_boundary.flip(dims=[0])[:-1, :self.dim],
                left_boundary.flip(dims=[0])[:-1, :self.dim],
                centerline.flip(dims=[0])[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                right_vectors[:, :2],
                                                                center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                        -left_vectors.flip(dims=[0])[:, :2],
                        -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                    dim=0)
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [-right_vectors.flip(dims=[0])[:, 2],
                -left_vectors.flip(dims=[0])[:, 2],
                -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            crosswalk_type = self._point_types.index('CROSSWALK')
            center_type = self._point_types.index('CENTERLINE')
            point_type[crosswalk_idx] = torch.cat([
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(left_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []


        for lane_segment_id, lane_segment in lanes.items():

            lane_segment_idx = polygon_ids.index(lane_segment_id)
            
            pred_inds = []
            for pred_id in lane_segment['predecessors']:
                pred_idx = safe_list_index(polygon_ids, pred_id)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))

            succ_inds = []
            for succ_id in lane_segment['successors']:
                succ_idx = safe_list_index(polygon_ids, succ_id)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))

            if lane_segment['l_neighbor_id'] != 'None':
                left_idx = safe_list_index(polygon_ids, lane_segment['l_neighbor_id'])
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))

            if lane_segment['r_neighbor_id'] != 'None':
                right_idx = safe_list_index(polygon_ids, lane_segment['r_neighbor_id'])
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))

        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data



    def get_agent_features(self, df, start_idx, num_steps) -> Dict[str, Any]:
        """
        dfから取り出すのは以下の三つ
        1. timestamp
        2. id
        3. x, y
        """
        t = np.array(sorted(list(set(df['timestamp'].values))))
        t = t[start_idx:start_idx + num_steps]

        agent_ids_all = list(set(df['id'].values))

        agent_ids = []
        for agent_id in agent_ids_all:
            agent_df = df.loc[df['id'] == agent_id]
            agent_timestamps = agent_df['timestamp'].values
            if np.sum([timestamp in t for timestamp in agent_timestamps]) > 2:
                agent_ids.append(agent_id)


        num_agents = len(agent_ids)
        av_idx = 0

        # initialization
        valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, num_steps, self.dim, dtype=torch.float)

        for agent_idx, aid in enumerate(agent_ids):
            
            agent_df = df.loc[df['id'] == aid]
            agent_timestamps = agent_df['timestamp'].values

            agent_steps = np.zeros_like(agent_timestamps, dtype=bool)
            for i,timestamp in enumerate(agent_timestamps):
                if timestamp in t:
                    agent_steps[i] = True

            agent_xy = agent_df[['x', 'y']].values[agent_steps]
            
            agent_steps = np.zeros(num_steps, dtype=bool)
            for i,timestamp in enumerate(t):
                if timestamp in agent_timestamps:
                    agent_steps[i] = True

            agent_heading = np.arctan2(agent_xy[1:,1]-agent_xy[:-1,1], agent_xy[1:,0]-agent_xy[:-1,0])
            agent_heading = np.hstack([agent_heading, [agent_heading[-1]]])
            agent_velocity_x = np.diff(agent_xy[:,0]) / (np.diff(t[agent_steps[:len(t)]])+1e-10)
            agent_velocity_x = np.hstack([agent_velocity_x, agent_velocity_x[-1]])
            agent_velocity_y = np.diff(agent_xy[:,1]) / (np.diff(t[agent_steps[:len(t)]])+1e-10)
            agent_velocity_y = np.hstack([agent_velocity_y, agent_velocity_y[-1]])
        
            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = aid
            agent_type[agent_idx] = self._agent_types.index(self.v2x_to_qcnet_agent_types(agent_df['type'].values[0]))
            agent_category[agent_idx] = 2
            position[agent_idx, agent_steps, :2] = torch.from_numpy(agent_xy).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(agent_heading).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([agent_velocity_x,agent_velocity_y],axis=-1)).float()

        predict_mask[current_valid_mask
                        | (agent_category == 2)
                        | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
        }


def qcnet_pred(model, data, num_historical_steps=50, device='cuda'):
    """
    Input
        model <model.QCNet.QCNet>

        data <torch_geometric.data.hetero_data.HeteroData>
    """
    
    with torch.no_grad(): 
        pred = model(data.to(device))

    traj_refine = torch.cat([pred['loc_refine_pos'][..., :2], pred['scale_refine_pos'][..., :2]], dim=-1)
    pi = pred['pi']

    eval_mask = data['agent']['category'] == 2
    origin_eval = data['agent']['position'][eval_mask, num_historical_steps - 1]
    theta_eval = data['agent']['heading'][eval_mask, num_historical_steps - 1]
    cos, sin = theta_eval.cos(), theta_eval.sin()
    rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device='cuda')
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
    pi_eval = F.softmax(pi[eval_mask], dim=-1)

    traj_eval = traj_eval.cpu().numpy()
    pi_eval = pi_eval.cpu().numpy()

    return traj_eval, pi_eval, pred



def create_montage(image_arrays, grid_size):
    rows, cols = grid_size
    
    # Get the size of the first image
    tile_height, tile_width, _ = image_arrays[0].shape
    
    # Create a blank canvas for the montage
    montage_height = rows * tile_height
    montage_width = cols * tile_width
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)+255
    
    for idx, img_array in enumerate(image_arrays):
        if idx >= rows * cols:
            break
        
        # Calculate position in the grid
        row = idx // cols
        col = idx % cols
        y = row * tile_height
        x = col * tile_width
        
        # Place the image on the montage canvas
        montage[y:y+tile_height, x:x+tile_width] = img_array
    
    return montage