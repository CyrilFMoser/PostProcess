from operator import mul
import os
import shutil 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
from sklearn.neighbors import KDTree
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import pandas as pd
from scipy.spatial import cKDTree
import time
from data.collate.collate import point_collate_fn
from torch_cluster import knn
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import sys
from queue import Queue
from threading import Lock
import torch.distributed as dist


class ScanNetPPV2DataSet(Dataset):
    """
    Dataset that loads the per gaussian labels derived from the point cloud labels provided in the Scannetppv2 dataset
    """

    def __init__(self,  scene_root, metadata_root,device="cpu",mode="train",subset=None,filtered_scenes=None,class_subset=100,sort_classes=True,subset_cache_root=None):
        """
        filtered_scenes: list of scene identifiers to filter out (str)
        scene_root: root folder where all files defining the 3D gaussian splats are stored
        feature_root: root folder where all files storing the per gaussian features of scenes are stored
        metadata_root: root folder where all files storing metadata of this dataset are stored
        device: device used for precomputing
        mode: train or validation
        subset: number of Gaussians per box. If None, use all
        class_subset: k number of top-k classes to use. 100 if None
        sort_classes: sort_classes by occurence per box
        subset_cache_root: root where potentially already precomputed subset are
        """            
        if mode == "train":
            self.scene_root = os.path.join(scene_root,"train_grid1.0cm_chunk6x6_stride3x3")
        elif mode == "validation":
            self.scene_root = os.path.join(scene_root,"val")
            if not os.path.exists(self.scene_root):
                self.scene_root = os.path.join(scene_root,"train_grid1.0cm_chunk6x6_stride3x3")
                print("Using training dataset as validation because val set doesn't exist!")
        self.metadata_root = metadata_root
        self.mode = mode
        self.scene_names = os.listdir(self.scene_root) # ["0a5c013435_0","0a7cc12c0e_0","0a7cc12c0e_1"]
        if filtered_scenes is not None:
            for scene in filtered_scenes:
                self.scene_names.remove(scene)

        self.text_embeddings_file = os.path.join(metadata_root,"scanetpp100_text_embeddings_siglip2.pt")
        self.sort_classes = sort_classes
        self.device = device

        self.collate_fn = point_collate_fn

        self.num_classes = class_subset

        self.num_channels = self.num_classes + 11
        self.subset = subset
        self.ignore_index = -1

        self.overlap = 0.05

        if self.subset is not None and subset_cache_root is not None:
            self.subset_cache_root = os.path.join(subset_cache_root,os.path.basename(self.scene_root))
            self.box_config = f"boxes_N{int(self.subset/1000)}k_p{int(self.overlap*100)}"
            self.preprocess_subsets()

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        start_total = time.time()
        scene_name,box_name = self.boxes[idx]
        small = box_name == "full" # scene didn't need to be subsampled, but we still have 25-NN queries
        #print(scene_name)
        data = {}
        data["name"] = scene_name

        scene_dir = os.path.join(self.scene_root, scene_name)
        coord_file = os.path.join(scene_dir, "coord.npy")
        color_file = os.path.join(scene_dir,"color.npy") # 3
        quat_file = os.path.join(scene_dir,"quat.npy") # 4
        scale_file = os.path.join(scene_dir,"scale.npy") # 3
        opacity_file = os.path.join(scene_dir,"opacity.npy") # 1
        feat_file = os.path.join(scene_dir,"lang_feat.npy")
        segment_file = os.path.join(scene_dir,"segment.npy")
        pc_coord_file = os.path.join(scene_dir,"pc_coord.npy")
        pc_segment_file = os.path.join(scene_dir,"pc_segment.npy")
        valid_feat_mask_file = os.path.join(scene_dir,"valid_feat_mask.npy")
        logits_file = os.path.join(scene_dir,"logits.pt")
        neighbours_file = os.path.join(self.subset_cache_root,self.box_config,scene_name,box_name,f"neighbours.npy")
        if not small:
            pc_mask_file = os.path.join(self.subset_cache_root,self.box_config,scene_name,box_name,"pc_subset_mask.npy")
            gs_mask_file = os.path.join(self.subset_cache_root,self.box_config,scene_name,box_name,"gs_subset_mask.npy")
        start_load = time.time()
        coord = np.load(coord_file)
        data["coord"] = torch.Tensor(coord)
        pc_coord = np.load(pc_coord_file)
        data["pc_coord"] = torch.Tensor(pc_coord)
        subset_mask = None
        start = time.time()
        if self.subset is not None and not small:
            subset_mask = np.load(gs_mask_file)
            pc_subset_mask = np.load(pc_mask_file)
        end = time.time()
        subsetting = end-start
        feat = None
        if os.path.exists(logits_file):
            try:
                feat = torch.load(logits_file)
            except:
                print(f"Failed to load logits_file for {idx}")
                pass
        if feat is None: # never calculated this scene's logits before
            device = torch.device("cuda",idx%4)
            feat = torch.Tensor(np.load(feat_file)).to(device)
            
            norm = feat.norm(dim=-1, keepdim=True)
            feat.div_(norm.clamp_min_(1e-12))
            text_embeddings = torch.load(self.text_embeddings_file,weights_only=True).to(device)
            text_embeddings = F.normalize(text_embeddings, dim=1)
            feat = (feat @ text_embeddings.T).cpu()  # cosine similarity
            del text_embeddings
            torch.save(feat,logits_file)

        if self.subset is not None and not small:
            feat = feat[subset_mask]
        end_load = time.time()
        load_subset = end_load - start_load - subsetting
        start = time.time()
        nc = feat.shape[1] # full number of classes
        base_pred = feat.argmax(dim=1)
        counts = torch.bincount(base_pred,minlength=nc)
        _, sorted_mask = torch.sort(counts,descending=True)
        feat = feat[:,sorted_mask]
        data["full_sorted_mask"]=sorted_mask
        # potentially subset number of classes here:
        feat = feat[:,:self.num_classes]
        end = time.time()
        sorting = end-start
        start = time.time()
        if self.subset is not None and not small:
            gs_features = [torch.Tensor(np.load(file))[subset_mask] for file in [color_file,quat_file,scale_file]]
            gs_features.append(torch.Tensor(np.load(opacity_file)[subset_mask]).unsqueeze(1))
        else:
            gs_features = [torch.Tensor(np.load(file)) for file in [color_file,quat_file,scale_file]]
            gs_features.append(torch.Tensor(np.load(opacity_file)).unsqueeze(1))
        feat_list = [feat]
        feat_list.extend(gs_features)
        feat = torch.cat(feat_list,dim=1)
        data["feat"] = feat
        del feat_list
        end = time.time()
        feat_construction = end-start

        start = time.time()
        N = data["coord"].shape[0]
        #data["batch"] = torch.zeros(N, dtype=torch.long).to(self.device)
        data["grid_size"] = torch.Tensor([0.01])
        gs_labels = torch.Tensor(np.load(segment_file)).long()[:,0]
        valid_mask = torch.Tensor(gs_labels != self.ignore_index)
        valid_feat_mask = torch.Tensor(np.load(valid_feat_mask_file)).to(bool)
        data["valid_feat_mask"] = valid_feat_mask

        pc_labels = torch.Tensor(np.load(pc_segment_file)).long()[:,0]
        
        
        data["ignore_index"] = self.ignore_index
        data["neighbours"] = torch.Tensor(np.load(neighbours_file)).long()
        end = time.time()
        various_loading = end-start

        # reorder the labels according to the rest
        start = time.time()
        mapping = torch.empty_like(sorted_mask)
        mapping[sorted_mask] = torch.arange(nc)
        gs_labels[valid_mask] = mapping[gs_labels[valid_mask]]
        data["segment"] = gs_labels
        data["valid_mask"] = valid_mask & (data["segment"] < self.num_classes) & valid_feat_mask
        # reorder class order in pointclouds
        pc_valid_mask = pc_labels != self.ignore_index
        pc_labels[pc_valid_mask] = mapping[pc_labels[pc_valid_mask]]
        # ignore points that expect a class we don't have
        pc_labels[pc_labels >= self.num_classes] = self.ignore_index
        data["pc_segment"] = pc_labels
        end = time.time()
        reorder = end-start

        start = time.time()
        if self.subset is not None and not small:
            for key in ["segment","valid_mask","valid_feat_mask","coord"]:
                data[key] = data[key][subset_mask]
            for key in ["pc_segment","pc_coord"]:
                data[key] = data[key][pc_subset_mask]
        end = time.time()
        point_subset = end-start
        data["pc_offset"] = torch.Tensor([data["pc_coord"].shape[0]]).long()
        data["offset"] = torch.Tensor([data["coord"].shape[0]]).long()
        end_total = time.time()
        #print(f"\n Total Time: {end_total-start_total:.3f}s | PointSubset Time: {point_subset:.3f}s | Reorder Time: {reorder:.3f}s | Feat Construction: {feat_construction:.3f}s | Subsetting: {subsetting:.3f}s | Sorting: {sorting:.3f}s | Load Subset: {load_subset:.3f}s | Various Loading: {various_loading:.3f}s")
        return data
    
    def subset_scene_old(self,coord):
        mask = np.ones(len(coord), dtype=bool)
        if len(mask) <=self.subset:
            return mask 
        mid_y = 0.5
        x_min = coord[:,0].min()
        x_max = coord[:,0].max()
        y_min = coord[:,1].min()
        y_max = coord[:,1].max()
        while mid_y >= 0.1:
            mid_x = 0.5
            while mid_x >= 0.1:
                for x_smaller in [0,1,2]:
                    if x_smaller==0: # x <= x_mid
                        x_mask = coord[:,0] <= (x_min + (x_max - x_min) * mid_x) 
                    elif x_smaller == 1: # x > x_mid
                        x_mask = coord[:,0] >  (x_max - (x_max - x_min) * mid_x)
                    else:
                        x_mask = np.ones(len(coord), dtype=bool)
                    for y_smaller in [0,1,2]:
                        if y_smaller==0: # y < y_mid
                            y_mask = coord[:,1] <= (y_min + (y_max - y_min) * mid_y) 
                        elif y_smaller==1: # y > y_mid
                            y_mask = coord[:,1] > (y_max - (y_max - y_min) * mid_y)
                        elif x_smaller==2: # can't ignore both
                            continue
                        else:
                            y_mask = np.ones(len(coord), dtype=bool)
                        cur_mask = x_mask & y_mask
                        if self.subset * 0.6 <= cur_mask.sum() < self.subset:
                            if self.mode=="train" and np.random.random(1)[0] <= 0.5:
                                return cur_mask
                            else:
                                mask = cur_mask
                                continue
                        if cur_mask.sum() < self.subset and cur_mask.sum() > mask.sum():
                            mask = cur_mask
                mid_x-=0.1
            mid_y -=0.1
        return mask
    
    def preprocess_subsets(self):
        rank = dist.get_rank()
        is_main_process = dist.get_rank() == 0

        self.cache = os.path.join(self.subset_cache_root,self.box_config)
        os.makedirs(self.cache,exist_ok=True)
        os.makedirs(os.path.join(self.metadata_root,os.path.basename(self.scene_root)),exist_ok=True)
        csv_file = os.path.join(self.metadata_root,os.path.basename(self.scene_root),f"{self.box_config}.csv")
        columns = ["Scene", "Boxes", "Files"]

        # Check if the file exists and is not empty
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            # Load the CSV normally
            df = pd.read_csv(csv_file)
        else:
            # Initialize an empty DataFrame with headers
            df = pd.DataFrame(columns=columns)
        df.set_index("Scene",inplace=True)

        if is_main_process:
            log_dir = Path("logs")
            timing_logger = logging.getLogger("preprocess_timing_logger")
            timing_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_dir / "preprocess_timing.log",mode="w")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(formatter)

            timing_logger.addHandler(fh)
            timing_logger.propagate = False 
        # Need to know which index corresponds to which box:
        self.boxes = []
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus))  # e.g., [0, 1, 2, 3]
        num_workers = num_gpus

        progress_bar = tqdm(total=len(self.scene_names))
        progress_lock = Lock()
        logger_lock = Lock()
        df_lock = Lock()
        scene_queue = Queue()
        for step, scene in enumerate(self.scene_names):
            scene_queue.put((step, scene))

        def gpu_worker(device_id):
            torch.cuda.set_device(device_id)
            results = []
            text_embeddings = torch.load(self.text_embeddings_file,weights_only=True).to(f"cuda:{device_id}")
            text_embeddings = F.normalize(text_embeddings, dim=1)
            while not scene_queue.empty():
                try:
                    step, scene = scene_queue.get_nowait()
                except Exception:
                    break

                boxes, df_row, logs = preprocess_scene(
                    scene, step, df, self.cache, self.subset,
                    self.overlap, f"cuda:{device_id}", self.scene_root,self.ignore_index,text_embeddings
                )
                results.append((boxes, df_row))
                with progress_lock:
                    progress_bar.update(1)
                if logs is not None and is_main_process:
                    with logger_lock:
                        timing_logger.info(logs)
                if df_row is not None and is_main_process:
                    with df_lock:
                        df.loc[boxes[0][0]] = df_row
                        df.reset_index().to_csv(csv_file,index=False)
            return results
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            try:
                futures = [executor.submit(gpu_worker, device_id) for device_id in device_ids]
                for future in as_completed(futures):
                    try:
                        for boxes, df_row in future.result():
                            self.boxes.extend(boxes)
                    except Exception as e:
                        print(f"Scene failed: {e}")
                self.boxes.sort(key=lambda x: (x[0], x[1]))  # deterministic: (scene_name, box_name)
            except KeyboardInterrupt:
                print("\n🛑 Ctrl+C detected — shutting down workers...", flush=True)

                # Cancel tasks that haven't started
                executor.shutdown(wait=False, cancel_futures=True)

                # Kill remaining worker processes
                for p in mp.active_children():
                    print(f"Terminating worker PID {p.pid}", flush=True)
                    p.terminate()
                sys.exit(1)
                            
                 
def preprocess_scene(scene,step,df,cache,subset,overlap,device,scene_root,ignore_index,text_embeddings):
    scene_cache = os.path.join(cache,scene)
    start = time.time()
    files_per_box = 3
    files_per_box_small = 1
    scene_dir = os.path.join(scene_root,scene)
    feat_file = os.path.join(scene_dir,"lang_feat.npy")        
    logits_file = os.path.join(scene_dir,"logits.pt")
    if not os.path.exists(logits_file):
        feat = torch.Tensor(np.load(feat_file)).to(device)
        norm = feat.norm(dim=-1, keepdim=True)
        feat.div_(norm.clamp_min_(1e-12))
        feat = (feat @ text_embeddings.T).cpu()  # cosine similarity
        torch.save(feat,logits_file)
    #print("Starting processing",flush=True)
    if scene in df.index: #check if already processed
        # need to check if its still in scratch and fully intact
        row = df.loc[scene]
        num_boxes = row["Boxes"] # supposed number of boxes
        if os.path.exists(scene_cache) and len(os.listdir(scene_cache)) == num_boxes:
            num_files = row["Files"] # number of files per box
            if num_boxes == 1:
                files_per_box = files_per_box_small # we have a small scene
            redo = num_files != files_per_box
            if not redo:
                for box in os.listdir(scene_cache):
                    if num_files!=len(os.listdir(os.path.join(scene_cache,box))):
                        redo = True
                        break
            if not redo:
                if num_boxes == 1:
                    names = [(scene,"full")]
                else:
                    names = [(scene,f"box{i}") for i in range(num_boxes)]
                return names, None, f"Scene {step:05d} | Precheck: {time.time()-start:.3f}s (Successful)"
        shutil.rmtree(scene_cache) # something was messed up
    elif os.path.exists(scene_cache):
        shutil.rmtree(scene_cache) # something was messed up
    end = time.time()
    precheck = end-start
    os.makedirs(scene_cache,exist_ok=True)
    scene_dir = os.path.join(scene_root, scene)
    gs_coord_file = os.path.join(scene_dir, "coord.npy")
    pc_coord_file = os.path.join(scene_dir, "pc_coord.npy")
    coord = torch.Tensor(np.load(gs_coord_file))
    N = coord.shape[0]
    #if N <= subset: # No need to subset
    #    return [(scene,None)], None, f"Scene {step:05d} | Precheck: {precheck:.3f}s | N: {N/1000:.1f}k"
    pc_coord = torch.Tensor(np.load(pc_coord_file)).to(device)
    coord = coord.to(device)
    
    valid_feat_mask_file = os.path.join(scene_dir,"valid_feat_mask.npy")
    valid_feat_mask = torch.Tensor(np.load(valid_feat_mask_file)).to(device).to(bool)
    pc_segment_file = os.path.join(scene_dir,"pc_segment.npy")
    pc_segment = torch.Tensor(np.load(pc_segment_file)).to(device).long()[:,0]
    pc_valid_mask = pc_segment != ignore_index
    
    start = time.time()
    row, col = knn(coord, pc_coord, k=1)
    orig_dists = (pc_coord[row] - coord[col]).norm(dim=1)
    max_allowed_dist = orig_dists.max()
    end = time.time()
    orig_knn = end-start
    start = time.time()
    boxes = split_xyz_overlap_torch(coord,subset,overlap)
    end = time.time()
    box_splitting = end-start
    num_boxes = len(boxes)
    if num_boxes == 1:
        files_per_box = files_per_box_small
    df_row = {"Boxes":num_boxes,"Files":files_per_box}
    pc_time = 0
    nn25_time = 0
    for i,gs_mask in enumerate(boxes):
        # get the subsetmask for the pointcloud
        start = time.time()
        if num_boxes > 1:
            box_coords = coord[gs_mask]
            row, col = knn(box_coords, pc_coord, k=1)
            new_dists = (pc_coord[row] - box_coords[col]).norm(dim=1)
            
            pc_mask = new_dists <= max_allowed_dist
            box_pc_coords = pc_coord[pc_mask]
            box_pc_coords = box_pc_coords#[pc_valid_mask[pc_mask]]
            gs_valid_feat_mask = valid_feat_mask[gs_mask]
        else:
            # small scene
            box_coords = coord
            box_pc_coords = pc_coord#[pc_valid_mask]
            gs_valid_feat_mask = valid_feat_mask
        end = time.time()
        pc_time += end-start
        # Get 25-nn
        start = time.time()
        num_query = box_pc_coords.shape[0]
        edge_index = knn(box_coords[gs_valid_feat_mask],box_pc_coords,k=25)
        neighbours = edge_index[1].view(num_query,25)
        end = time.time()
        nn25_time += end-start

        # save everything
        if num_boxes>1:
            pc_mask = pc_mask.cpu().numpy()
            gs_mask = gs_mask.cpu().numpy()
        neighbours = neighbours.cpu().numpy()
        if num_boxes >1:
            box_dir = os.path.join(scene_cache,f"box{i}")
        else:
            box_dir = os.path.join(scene_cache,f"full")
        os.makedirs(box_dir,exist_ok=True)
        if num_boxes>1:
            gs_subset_mask_file = os.path.join(box_dir,f"gs_subset_mask.npy")
            pc_subset_mask_file = os.path.join(box_dir,f"pc_subset_mask.npy")
        neighbours_file = os.path.join(box_dir,f"neighbours.npy")

        if num_boxes>1:
            np.save(gs_subset_mask_file,gs_mask)
            np.save(pc_subset_mask_file,pc_mask)
        np.save(neighbours_file,neighbours)
    logs = f"Scene {step:05d} | Precheck: {precheck:.3f}s | N_GS: {N/1000:.1f}k | N_PC: {pc_coord.shape[0]/1000:.1f}k | Orig-KNN: {orig_knn:.3f}s | Box Splitting: {box_splitting:.3f}s | PC-Time: {pc_time:.3f}s | NN-25: {nn25_time:.3f}s"
    if num_boxes == 1:
        names = [(scene,"full")]
    else:
        names = [(scene,f"box{i}") for i in range(num_boxes)]
    return names, df_row, logs


def split_xyz_overlap_torch(points, N, overlap, indices=None, root_N=None):
    """
    Recursively split 3D points into overlapping spatial boxes with <= N points.

    Args:
        points   : (M, 3) torch tensor (subset of original points)
        N        : max number of points per box
        overlap  : fraction (0–0.5 recommended) of points to overlap
        indices  : (M,) tensor of original indices corresponding to `points`
        root_N   : total number of points in the original point cloud

    Returns:
        List of masks: (root_N,) bool tensor
    """

    device = points.device

    # Top-level initialization
    if indices is None:
        indices = torch.arange(points.shape[0], device=device)
    if root_N is None:
        root_N = points.shape[0]

    M = points.shape[0]

    # Compute bounding box for this node
    bbox_min = points.min(axis=0).values
    bbox_max = points.max(axis=0).values

    # Base case: small enough
    if M <= N:
        mask = torch.zeros(root_N, dtype=torch.bool, device=device)
        mask[indices] = True
        return [mask]

    # Choose split axis (longest spatial dimension)
    axis_lengths = bbox_max - bbox_min
    split_axis = torch.argmax(axis_lengths).item()

    # Sort along split axis
    sorted_order = torch.argsort(points[:, split_axis])
    sorted_points = points[sorted_order]
    sorted_indices = indices[sorted_order]

    median_idx = M // 2
    overlap_n = int(M * overlap)

    # Clamp overlap so we don't collapse the split
    overlap_n = min(overlap_n, median_idx - 1) if median_idx > 1 else 0

    left_end = median_idx + overlap_n
    right_start = median_idx - overlap_n

    # Degenerate safety check
    if right_start <= 0 or left_end >= M:
        mask = torch.zeros(root_N, dtype=torch.bool, device=device)
        mask[indices] = True
        return [mask]

    # Split
    left_points = sorted_points[:left_end]
    left_indices = sorted_indices[:left_end]

    right_points = sorted_points[right_start:]
    right_indices = sorted_indices[right_start:]

    # Recurse
    left_boxes = split_xyz_overlap_torch(
        left_points, N, overlap, left_indices, root_N
    )
    right_boxes = split_xyz_overlap_torch(
        right_points, N, overlap, right_indices, root_N
    )

    return left_boxes + right_boxes