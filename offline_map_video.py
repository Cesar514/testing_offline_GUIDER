#!/usr/bin/env python3

import os
import sys
import math
import argparse
from pathlib import Path
import time
import numpy as np
from collections import deque
import cv2

# ROS 2 bag reading
import rosbag2_py
from rclpy.serialization import deserialize_message

# ROS messages
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

###############################################################################
# 1) Offline Belief Map Class (same as before)
###############################################################################
class OfflineBeliefMap:
    def __init__(self):
        # Map data
        self.occupancy   = None  # shape (H,W), int8
        self.width       = 0
        self.height      = 0
        self.resolution  = 0.05
        self.origin_x    = 0.0
        self.origin_y    = 0.0

        # Belief fields
        self.base_belief_map    = None
        self.motion_layer       = None
        self.synergy_layer      = None
        self.object_static_mask = None

        # For computing velocity & threshold
        self.last_update_pose       = None
        self.last_odom_for_velocity = None
        self.update_threshold       = 0.3

        # Inflation & synergy BFS
        self.inflation_radius_cells = 15
        self.prob_at_center = 0.6
        self.prob_at_edge   = 0.2

        # Motion Horizons
        self.horizons      = [5.0, 10.0, 30.0]
        self.prediction_dt = 0.1

        # Decay/Blend
        self.motion_decay_rate     = 0.25
        self.motion_blend_rate     = 0.4
        self.base_decay_rate_free  = 0.55
        self.base_decay_rate_obj   = 0.50

        # Synergy BFS
        self.synergy_decay_rate   = 0.75
        self.synergy_inflate_dist = 0.15
        self.synergy_lower_bound  = 0.6
        self.synergy_set_value    = 0.7
        self.synergy_increment    = 0.05
        self.synergy_cap          = 0.90

    def set_map(self, occ_array, width, height, resolution, origin_x, origin_y):
        self.occupancy  = occ_array
        self.width      = width
        self.height     = height
        self.resolution = resolution
        self.origin_x   = origin_x
        self.origin_y   = origin_y

        outside_mask = self.find_outside_regions(occ_array)
        base_map = np.full((height, width), -1.0, dtype=np.float32)
        base_map[outside_mask] = 0.0

        # free => 0.02
        free_mask = ((occ_array == 0) & (~outside_mask))
        base_map[free_mask] = 0.02

        # Object inflation
        self.object_static_mask = np.zeros((height, width), dtype=bool)
        occupied_clusters = self.cluster_occupied_cells(occ_array)
        for cluster in occupied_clusters:
            if not self.cluster_touches_outside(cluster, outside_mask):
                self.inflate_cluster(cluster, occ_array, base_map, outside_mask)

        obj_mask = (base_map >= 0.2)
        self.object_static_mask[obj_mask] = True

        # synergy + motion init
        motion_map = np.zeros((height, width), dtype=np.float32)
        motion_map[outside_mask] = -1.0
        synergy_map = np.zeros((height, width), dtype=np.float32)
        synergy_map[outside_mask] = -1.0

        self.base_belief_map = base_map
        self.motion_layer    = motion_map
        self.synergy_layer   = synergy_map

    def odom_update(self, t, x, y):
        if self.base_belief_map is None:
            return

        if self.last_update_pose is None:
            self.last_update_pose = (t, x, y)
            self.last_odom_for_velocity = (t, x, y)
            return
        if self.last_odom_for_velocity is None:
            self.last_odom_for_velocity = (t, x, y)
            return

        old_t, old_x, old_y = self.last_odom_for_velocity
        dt = t - old_t
        if dt <= 0:
            return

        vx = (x - old_x)/dt
        vy = (y - old_y)/dt
        self.last_odom_for_velocity = (t, x, y)

        upd_t, upd_x, upd_y = self.last_update_pose
        dist_moved = math.hypot(x - upd_x, y - upd_y)
        if dist_moved > self.update_threshold:
            # 1) Decay
            self.decay_base_belief()
            self.decay_motion_layer()
            self.decay_synergy_layer()

            # 2) Multi-horizon evidence
            all_ev = np.zeros_like(self.base_belief_map, dtype=np.float32)
            for i, horizon in enumerate(self.horizons):
                pts = self.predict_trajectory(x, y, vx, vy, horizon, self.prediction_dt)
                ev  = self.compute_motion_evidence(pts, 1.0)
                if i < len(self.horizons) - 1:
                    ev = np.minimum(ev, 0.6)
                else:
                    ev = np.minimum(ev, 0.85)
                all_ev = np.maximum(all_ev, ev)

            self.blend_motion_layer(all_ev)
            self.update_synergy(all_ev)

            self.last_update_pose = (t, x, y)

    def get_combined_belief(self):
        if self.base_belief_map is None:
            return None
        c1 = np.maximum(self.base_belief_map, self.motion_layer)
        return np.maximum(c1, self.synergy_layer)

    # -------------------- BFS synergy logic / motion decay --------------------
    def decay_base_belief(self):
        if self.base_belief_map is None or self.object_static_mask is None:
            return
        known_mask = (self.base_belief_map >= 0.0)
        if not np.any(known_mask):
            return

        # free => decays to near zero
        free_mask = known_mask & (~self.object_static_mask)
        self.base_belief_map[free_mask] *= self.base_decay_rate_free
        negative_mask = (self.base_belief_map >= 0.0) & (self.base_belief_map < 0.001)
        self.base_belief_map[negative_mask] = 0.0

        # objects => partial decay => min 0.1
        obj_mask = known_mask & self.object_static_mask
        self.base_belief_map[obj_mask] *= self.base_decay_rate_obj
        below_obj = obj_mask & (self.base_belief_map < 0.1)
        self.base_belief_map[below_obj] = 0.1

    def decay_motion_layer(self):
        if self.motion_layer is None:
            return
        inside_mask = (self.motion_layer >= 0.0)
        self.motion_layer[inside_mask] *= self.motion_decay_rate

    def decay_synergy_layer(self):
        if self.synergy_layer is None:
            return
        inside_mask = (self.synergy_layer >= 0.0)
        self.synergy_layer[inside_mask] *= self.synergy_decay_rate

    def predict_trajectory(self, x, y, vx, vy, horizon, dt):
        points = []
        steps = int(horizon / dt)
        rx, ry = x, y
        for _ in range(steps):
            rx += vx*dt
            ry += vy*dt
            points.append((rx, ry))
        return points

    def compute_motion_evidence(self, predicted_points, max_dist_m=1.0):
        if self.base_belief_map is None:
            return None
        evid = np.zeros((self.height, self.width), dtype=np.float32)
        for (px, py) in predicted_points:
            col = int((px - self.origin_x)/self.resolution)
            row = int((py - self.origin_y)/self.resolution)
            rad_cells = int(max_dist_m/self.resolution)

            for rr in range(row - rad_cells, row + rad_cells + 1):
                for cc in range(col - rad_cells, col + rad_cells + 1):
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        dx = (cc - col)*self.resolution
                        dy = (rr - row)*self.resolution
                        dist = math.hypot(dx, dy)
                        if dist <= max_dist_m:
                            val = 1.0 - dist/max_dist_m
                            evid[rr, cc] = max(evid[rr, cc], val)
        return evid

    def blend_motion_layer(self, new_ev):
        if self.motion_layer is None or new_ev is None:
            return
        inside_mask = (self.motion_layer >= 0.0)
        self.motion_layer[inside_mask] = np.clip(
            self.motion_layer[inside_mask] + self.motion_blend_rate* new_ev[inside_mask],
            0.0, 1.0
        )

    def update_synergy(self, motion_evidence):
        if (self.synergy_layer is None or
            self.object_static_mask is None or
            motion_evidence is None):
            return
        intersection = self.object_static_mask & (motion_evidence > 0.01)
        rr_cc = np.argwhere(intersection)
        radius_cells = max(1, int(self.synergy_inflate_dist/self.resolution + 0.5))
        for (rr, cc) in rr_cc:
            self.inflate_synergy(rr, cc, radius_cells)

    def inflate_synergy(self, r, c, radius_cells):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        visited = set()
        queue = deque()
        queue.append((r, c, 0))
        visited.add((r, c))
        while queue:
            rr, cc, dist = queue.popleft()
            if self.synergy_layer[rr, cc] >= 0.0:
                old_val = self.synergy_layer[rr, cc]
                if old_val < self.synergy_lower_bound:
                    new_val = self.synergy_set_value
                else:
                    new_val = old_val + self.synergy_increment
                    if new_val > self.synergy_cap:
                        new_val = self.synergy_cap
                self.synergy_layer[rr, cc] = new_val

            if dist < radius_cells:
                for (dr, dc) in dirs:
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if (nr, nc) not in visited:
                            if self.synergy_layer[nr, nc] >= 0.0:
                                visited.add((nr, nc))
                                queue.append((nr, nc, dist+1))

    # BFS for outside + cluster occupied
    def find_outside_regions(self, grid):
        h, w = grid.shape
        outside_mask = np.zeros((h, w), dtype=bool)
        queue = deque()

        # BFS from edges
        for r in range(h):
            queue.append((r,0))
            queue.append((r,w-1))
        for c in range(w):
            queue.append((0,c))
            queue.append((h-1,c))

        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        while queue:
            rr, cc = queue.popleft()
            if not(0<=rr<h and 0<=cc<w):
                continue
            if outside_mask[rr, cc]:
                continue
            if grid[rr, cc] in [0, -1]:
                outside_mask[rr, cc] = True
                for (dr,dc) in dirs:
                    nr, nc = rr+dr, cc+dc
                    if 0<=nr<h and 0<=nc<w:
                        if not outside_mask[nr,nc]:
                            queue.append((nr,nc))
        return outside_mask

    def cluster_occupied_cells(self, grid):
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        clusters = []
        for r in range(h):
            for c in range(w):
                if grid[r,c] == 100 and not visited[r,c]:
                    cluster = []
                    stack = [(r,c)]
                    visited[r,c] = True
                    while stack:
                        rr, cc = stack.pop()
                        cluster.append((rr, cc))
                        for (dr, dc) in dirs:
                            nr, nc = rr+dr, cc+dc
                            if 0<=nr<h and 0<=nc<w:
                                if grid[nr,nc] == 100 and not visited[nr,nc]:
                                    visited[nr,nc] = True
                                    stack.append((nr,nc))
                    clusters.append(cluster)
        return clusters

    def cluster_touches_outside(self, cluster, outside_mask):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for (r, c) in cluster:
            for (dr,dc) in dirs:
                rr, cc = r+dr, c+dc
                if 0<=rr< outside_mask.shape[0] and 0<=cc< outside_mask.shape[1]:
                    if outside_mask[rr, cc]:
                        return True
        return False

    def inflate_cluster(self, cluster, grid, belief_map, outside_mask):
        visited = np.zeros_like(grid, dtype=bool)
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        queue = deque()
        for (r, c) in cluster:
            queue.append((r,c,0))
            visited[r,c] = True
        while queue:
            rr, cc, dist = queue.popleft()
            alpha = dist/ float(self.inflation_radius_cells)
            prob_here = self.prob_at_center + alpha*(self.prob_at_edge - self.prob_at_center)
            if belief_map[rr, cc] < prob_here:
                belief_map[rr, cc] = prob_here
            if dist< self.inflation_radius_cells:
                for (dr,dc) in dirs:
                    nr, nc = rr+dr, cc+dc
                    if 0<=nr< grid.shape[0] and 0<=nc< grid.shape[1]:
                        if not visited[nr,nc] and not outside_mask[nr,nc]:
                            visited[nr,nc] = True
                            queue.append((nr,nc, dist+1))

###############################################################################
# 2) Single-center hotspot BFS
###############################################################################
def find_hotspot_centers(synergy_map, threshold, rx, ry, origin_x, origin_y, resolution):
    """
    synergy>= threshold => BFS => contiguous => pick the nearest cell to (rx, ry)
    Return a list of (row, col) for the single center of each region.
    """
    if synergy_map is None:
        return []

    h, w = synergy_map.shape
    high_mask = (synergy_map >= threshold)
    visited = np.zeros_like(high_mask, dtype=bool)
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    centers = []

    for r in range(h):
        for c in range(w):
            if high_mask[r,c] and not visited[r,c]:
                stack = [(r,c)]
                visited[r,c] = True
                region = []
                while stack:
                    rr, cc = stack.pop()
                    region.append((rr,cc))
                    for (dr, dc) in directions:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w:
                            if high_mask[nr,nc] and not visited[nr,nc]:
                                visited[nr,nc] = True
                                stack.append((nr,nc))

                # pick the nearest cell to robot
                best_rc = None
                best_dist= 1e9
                for (rr, cc) in region:
                    wx = origin_x + cc*resolution
                    wy = origin_y + rr*resolution
                    d = math.hypot(wx - rx, wy - ry)
                    if d< best_dist:
                        best_dist = d
                        best_rc   = (rr, cc)
                if best_rc:
                    centers.append(best_rc)
    return centers

def draw_star(img, cx, cy, size=5, color=(0,255,255), thickness=2):
    # draws an 'X' shape star
    cv2.line(img, (cx-size, cy-size), (cx+size, cy+size), color, thickness)
    cv2.line(img, (cx-size, cy+size), (cx+size, cy-size), color, thickness)

###############################################################################
# 3) Synergy overlay in memory
###############################################################################
def render_frame_opencv(
    bg_img_uint8,
    synergy_map,
    resolution, origin_x, origin_y,
    robot_x, robot_y,
    robot_width, robot_height
):
    # synergy => red overlay
    H, W, _ = bg_img_uint8.shape
    final = bg_img_uint8.copy()

    synergy_clamped = np.clip(synergy_map, -1.0, 1.0)
    synergy_mask = np.where(synergy_clamped<0, 0, synergy_clamped)
    synergy_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    synergy_bgr[...,2] = (synergy_mask*255).astype(np.uint8)

    alpha_map = synergy_mask*0.5
    alpha_3d  = np.dstack([alpha_map, alpha_map, alpha_map])
    final_f   = final.astype(np.float32)
    synergy_f = synergy_bgr.astype(np.float32)
    blended   = final_f*(1.0-alpha_3d) + synergy_f*alpha_3d
    final     = blended.astype(np.uint8)

    # draw robot => rectangle
    c_rob = int((robot_x - origin_x)/resolution)
    r_rob = int((robot_y - origin_y)/resolution)
    rob_w_px = int(robot_width/resolution)
    rob_h_px = int(robot_height/resolution)

    rx0 = c_rob - rob_w_px//2
    rx1 = rx0 + rob_w_px
    ry0 = r_rob - rob_h_px//2
    ry1 = ry0 + rob_h_px

    rx0 = max(0, min(W-1, rx0))
    rx1 = max(0, min(W-1, rx1))
    ry0 = max(0, min(H-1, ry0))
    ry1 = max(0, min(H-1, ry1))

    color_yellow=(0,255,255)
    cv2.rectangle(final, (rx0, ry0), (rx1, ry1), color_yellow, thickness=-1)

    return final

###############################################################################
# 4) Main script: no PNG, direct VideoWriter
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="""
        Offline synergy with direct video writing (no PNG).
        You can pass a bag path directly via --bag, or provide --experiment/--subtrial
        so the script resolves data/<exp>/<sub>/experiment_<exp>_subtrial_<sub>_0.db3 automatically.
    """)
    parser.add_argument("--bag", type=str, default=None,
                        help="Path to the bag (.db3 file or directory). Overrides --experiment/--subtrial.")
    parser.add_argument("--experiment", type=int, default=None,
                        help="Experiment (participant) number used to build the default bag path.")
    parser.add_argument("--subtrial", type=int, default=None,
                        help="Subtrial index used to build the default bag path.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory that contains experiment folders (default: <repo>/data).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Where to store outputs (default: alongside the bag).")
    parser.add_argument("--robot-width", type=float, default=0.5)
    parser.add_argument("--robot-height", type=float, default=0.4)
    parser.add_argument("--hotspot-threshold", type=float, default=0.85)
    parser.add_argument("--log-file", default="hotspots_log.txt")
    args = parser.parse_args()

    # Resolve bag path
    if args.bag:
        bag_path = Path(args.bag).expanduser().resolve()
        if bag_path.is_dir():
            # pick the first db3 inside directory
            db3_candidates = sorted(bag_path.glob("*.db3"))
            if not db3_candidates:
                sys.exit(f"[Error] No .db3 files found under {bag_path}")
            bag_path = db3_candidates[0]
    else:
        if args.experiment is None or args.subtrial is None:
            sys.exit("[Error] Provide --bag or both --experiment and --subtrial.")
        repo_root = Path(__file__).resolve().parents[2]
        data_root = Path(args.data_root).expanduser().resolve() if args.data_root else repo_root / "data"
        bag_dir = data_root / f"experiment_{args.experiment}_subtrial_{args.subtrial}"
        bag_path = bag_dir / f"experiment_{args.experiment}_subtrial_{args.subtrial}_0.db3"
        if not bag_path.exists():
            alt_path = bag_dir / f"experiment_{args.experiment}_subtrial_{args.subtrial}"
            if alt_path.exists():
                db3_candidates = sorted(alt_path.glob("*.db3"))
                if db3_candidates:
                    bag_path = db3_candidates[0]
        if not bag_path.exists():
            sys.exit(f"[Error] Bag not found at {bag_path}")

    # Resolve output directory
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = bag_path.parent / "guider_base"
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = str(outdir / args.log_file)

    # read the bag
    storage_opts = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_opts = rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                                 output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_opts, converter_opts)

    # Restrict to the topics we actually consume to avoid parsing large clouds/images.
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=["/map", "/odom", "/cmd_vel"]))
    except AttributeError:
        # Older rosbag2_py may not expose set_filter; proceed without filtering.
        pass

    belief = OfflineBeliefMap()
    have_map       = False
    t_cmdvel_start = None
    odom_msgs      = []

    while reader.has_next():
        topic_name, serialized_data, t_nanos = reader.read_next()
        t_s = float(t_nanos)*1e-9

        if topic_name=="/map" and not have_map:
            msg = deserialize_message(serialized_data, OccupancyGrid)
            occ = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
            belief.set_map(
                occ,
                msg.info.width,
                msg.info.height,
                msg.info.resolution,
                msg.info.origin.position.x,
                msg.info.origin.position.y
            )
            have_map = True

        elif topic_name=="/odom":
            msg = deserialize_message(serialized_data, Odometry)
            odom_msgs.append( (t_s,
                               msg.pose.pose.position.x,
                               msg.pose.pose.position.y) )

        elif topic_name=="/cmd_vel" and t_cmdvel_start is None:
            msg = deserialize_message(serialized_data, Twist)
            if (abs(msg.linear.x)>1e-9 or abs(msg.linear.y)>1e-9 or abs(msg.linear.z)>1e-9 or
                abs(msg.angular.x)>1e-9 or abs(msg.angular.y)>1e-9 or abs(msg.angular.z)>1e-9):
                t_cmdvel_start = t_s
                print(f"[INFO] First non-zero /cmd_vel at t={t_cmdvel_start:.2f}")

    odom_msgs.sort(key=lambda x: x[0])
    if not have_map:
        print("[Error] No /map => exit.")
        sys.exit(1)
    if not odom_msgs:
        print("[Error] No /odom => exit.")
        sys.exit(1)
    if t_cmdvel_start is None:
        print("[Warning] No non-zero /cmd_vel => no frames.")
        sys.exit(0)

    # background image
    H, W = belief.height, belief.width
    bg_img = np.ones((H, W, 3), dtype=np.uint8)*255
    free_mask = (belief.occupancy==0)
    bg_img[free_mask] = [190,190,190]
    wall_mask = (belief.occupancy==100)
    bg_img[wall_mask] = [50,50,50]
    # rest => white

    # filter odom
    valid_odom = [m for m in odom_msgs if m[0]>= t_cmdvel_start]
    if not valid_odom:
        print("[Warn] No odom after that => no frames.")
        sys.exit(0)

    # estimate fps from total duration
    total_duration = valid_odom[-1][0] - t_cmdvel_start
    if total_duration>0:
        fps = len(valid_odom)/ total_duration
    else:
        fps = 5.0

    # Some encoders require even resolution => fix if needed
    even_w = W if (W%2==0) else (W-1)
    even_h = H if (H%2==0) else (H-1)
    if even_w<2: even_w=2
    if even_h<2: even_h=2

    out_video_path = str(outdir / "synergy_opencv.mp4")

    # FourCC for H.264 can be 'avc1' or 'x264' in some builds, but let's try 'mp4v' for broad support
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(
        out_video_path,
        fourcc,
        fps,
        (even_w, even_h)
    )
    if not video_out.isOpened():
        print("[Error] Could not open VideoWriter => check codec/fourcc support.")
        sys.exit(1)

    # open log
    with open(log_path, "w") as f:
        f.write("# time(s), delta, (x1,y1) (x2,y2) ...\n")

    prev_hotset = set()
    wall_step_accum = 0.0
    max_wall_step = 0.0
    sim_step_accum = 0.0
    prev_sim_time = None
    processed_steps = 0

    for i,(t_s, rx, ry) in enumerate(valid_odom):
        compute_start = time.perf_counter()
        # synergy update
        belief.odom_update(t_s, rx, ry)
        synergy_map = belief.get_combined_belief()
        if synergy_map is None:
            continue

        # render synergy
        frame_bgr = render_frame_opencv(
            bg_img,
            synergy_map,
            belief.resolution,
            belief.origin_x, belief.origin_y,
            rx, ry,
            args.robot_width, args.robot_height
        )

        # find hotspots => draw
        rc_centers = find_hotspot_centers(
            synergy_map, args.hotspot_threshold,
            rx, ry,
            belief.origin_x, belief.origin_y,
            belief.resolution
        )
        new_hotspots=[]
        for (rr, cc) in rc_centers:
            draw_star(frame_bgr, cc, rr, size=5, color=(0,255,255), thickness=2)
            # world coords
            wx = round(belief.origin_x + cc*belief.resolution, 2)
            wy = round(belief.origin_y + rr*belief.resolution, 2)
            new_hotspots.append( (wx, wy) )

        # check if changed => log
        new_set = set(new_hotspots)
        if new_set!= prev_hotset:
            prev_hotset = new_set
            dt = t_s- t_cmdvel_start
            with open(log_path, "a") as f:
                coords_txt = " ".join(f"({x:.2f},{y:.2f})" for (x,y) in new_hotspots)
                f.write(f"{t_s:.2f}, {dt:.2f}, {coords_txt}\n")

        compute_elapsed = time.perf_counter() - compute_start
        wall_step_accum += compute_elapsed
        max_wall_step = max(max_wall_step, compute_elapsed)
        if prev_sim_time is not None:
            sim_step_accum += t_s - prev_sim_time
        prev_sim_time = t_s
        processed_steps += 1

        # possibly resize to even dims
        if (even_w!= W or even_h!= H):
            frame_bgr = cv2.resize(frame_bgr, (even_w, even_h), interpolation=cv2.INTER_AREA)

        video_out.write(frame_bgr)

        if i%200==0:
            print(f"[{i}/{len(valid_odom)}] frames => time={t_s:.2f}")

    video_out.release()
    print(f"Done. Wrote ~{len(valid_odom)} frames to {out_video_path}")
    print(f"Hotspots logged => {log_path}")

    total_compute = wall_step_accum
    sim_duration = total_duration if total_duration > 0 else 0.0
    avg_wall_step = wall_step_accum / processed_steps if processed_steps else 0.0
    avg_sim_step = sim_step_accum / (processed_steps-1) if processed_steps > 1 else 0.0
    rtf = (sim_duration / total_compute) if total_compute > 0 else float("inf")
    meets_one_hz = avg_wall_step < 1.0 if processed_steps else True

    perf_path = outdir / "performance_report.txt"
    with open(perf_path, "w") as pf:
        pf.write("pipeline\tguider_base\n")
        pf.write(f"frames_processed\t{processed_steps}\n")
        pf.write(f"sim_duration_s\t{sim_duration:.6f}\n")
        pf.write(f"compute_wall_s\t{total_compute:.6f}\n")
        pf.write(f"real_time_factor\t{rtf:.6f}\n")
        pf.write(f"avg_sim_step_s\t{avg_sim_step:.6f}\n")
        pf.write(f"avg_wall_step_s\t{avg_wall_step:.6f}\n")
        pf.write(f"max_wall_step_s\t{max_wall_step:.6f}\n")
        pf.write(f"avg_wall_step_lt_1s\t{int(meets_one_hz)}\n")
    print(f"Performance metrics saved ⇒ {perf_path}")

if __name__=="__main__":
    main()
