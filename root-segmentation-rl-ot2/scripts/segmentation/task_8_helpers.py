import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skan import Skeleton
from scipy.sparse.csgraph import dijkstra
from tensorflow.keras.models import load_model as keras_load_model

# === Configuration ===
MODEL_FILE = r"C:\Users\sasha\Documents\GitHub\2024-25b-fai2-adsai-SashaStacie230306\deliverables_tasks\sasha_230306_unet_model_v6_256px.h5"
PATCH_SIZE = 256

# === Load Model ===
def load_trained_network():
    return keras_load_model(MODEL_FILE, compile=False)

# === Image Preprocessing ===
def apply_edge_detection(gray_img):
    canny_edges = cv2.Canny(gray_img, 50, 150)
    _, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    return cv2.addWeighted(canny_edges, 0.6, binary_img, 0.4, 0)

def find_max_bounding_box(gray_img, edge_img):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, gray_img.shape[1], gray_img.shape[0]
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)

def crop_to_square(gray_img, bbox, pad_ratio=0.04):
    x, y, w, h = bbox
    side = max(w, h)
    center_x, center_y = x + w // 2, y + h // 2
    pad = int(side * pad_ratio)
    start_x = max(0, center_x - side // 2 - pad)
    start_y = max(0, center_y - side // 2 - pad)
    end_x = center_x + side // 2 + pad
    end_y = center_y + side // 2 + pad
    return gray_img[start_y:end_y, start_x:end_x]

# === Patch-based Mask Prediction ===
def predict_mask_from_patches(model, gray_img):
    border = 1
    padded_img = cv2.copyMakeBorder(gray_img, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=0)
    patch_list = []
    patch_coords = []

    for i in range(0, padded_img.shape[0], PATCH_SIZE):
        for j in range(0, padded_img.shape[1], PATCH_SIZE):
            patch = padded_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                temp = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                temp[:patch.shape[0], :patch.shape[1]] = patch
                patch = temp
            patch_list.append(patch)
            patch_coords.append((i, j))

    patches_array = np.array(patch_list)[..., np.newaxis] / 255.0
    predictions = model.predict(patches_array, verbose=0)

    full_mask = np.zeros_like(padded_img, dtype=np.float32)
    for idx, (i, j) in enumerate(patch_coords):
        h, w = padded_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE].shape
        full_mask[i:i+h, j:j+w] = predictions[idx].squeeze()[:h, :w]

    return full_mask[border:-border, border:-border]

# === Mask Post-processing ===
def refine_mask(seg_mask, area_threshold=75):
    binary_mask = (seg_mask > 0.5).astype(np.uint8)
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, struct_elem)

    labeled_mask = label(closed_mask)
    refined = np.zeros_like(closed_mask, dtype=np.uint8)
    for reg in regionprops(labeled_mask):
        if reg.area >= area_threshold:
            refined[labeled_mask == reg.label] = 1
    return refined

# === Root Length & Zone ===
def compute_skeleton_length(component_mask):
    skel_img = skeletonize(component_mask)
    if np.count_nonzero(skel_img) == 0:
        return 0.0

    skel_obj = Skeleton(skel_img)
    if skel_obj.n_paths == 0:
        return 0.0

    adj_matrix = (skel_obj.graph != 0).astype(int)
    degree_vals = adj_matrix.sum(axis=1).A.squeeze()
    endpoints = np.where(degree_vals == 1)[0]
    if len(endpoints) < 2:
        return 0.0

    coords = skel_obj.coordinates
    top_endpoint = min(endpoints, key=lambda idx: coords[idx, 0])
    distances, _ = dijkstra(csgraph=skel_obj.graph, directed=False,
                            return_predecessors=True, indices=top_endpoint)
    far_endpoint = max(endpoints, key=lambda idx: distances[idx])
    return distances[far_endpoint]

def get_top_five_roots(refined_mask, area_min=1, top_row_limit=0.3):
    labeled_mask = label(refined_mask > 0)
    img_h, img_w = refined_mask.shape
    candidates = []

    for reg in regionprops(labeled_mask):
        if reg.area < area_min:
            continue

        comp_mask = (labeled_mask == reg.label).astype(np.uint8)
        length_value = compute_skeleton_length(comp_mask)

        if reg.bbox[0] > top_row_limit * img_h:
            continue

        if length_value > 0.0:
            min_r, min_c, max_r, max_c = reg.bbox
            bbox = (min_c, min_r, max_c - min_c, max_r - min_r)
            candidates.append((bbox, length_value))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
    return candidates

def determine_zone(bbox, image_width, image_height):
    x, y, w, h = bbox
    zone_width = image_width // 5
    center_x = x + w / 2.0
    if center_x < zone_width:
        return 1
    elif center_x < 2 * zone_width:
        return 2
    elif center_x < 3 * zone_width:
        return 3
    elif center_x < 4 * zone_width:
        return 4
    else:
        return 5

# === Main Inference Function ===
def process_image(model, image_filepath, top_exclude=500, show_plots=False):
    base_label = os.path.splitext(os.path.basename(image_filepath))[0]

    img = cv2.imread(image_filepath)
    if img is None:
        raise FileNotFoundError(f"Unable to load {image_filepath}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges_img = apply_edge_detection(gray_img)
    bbox = find_max_bounding_box(gray_img, edges_img)
    cropped_img = crop_to_square(gray_img, bbox, pad_ratio=0.04)
    H, W = cropped_img.shape

    raw_mask = predict_mask_from_patches(model, cropped_img)
    refined_mask = refine_mask(raw_mask, area_threshold=75)
    refined_mask[:top_exclude, :] = 0

    top_five_roots = get_top_five_roots(refined_mask, area_min=1, top_row_limit=0.3)

    zone_length = {zone: 0.0 for zone in range(1, 6)}
    for bbox_val, length_val in top_five_roots:
        zone_id = determine_zone(bbox_val, W, H)
        if zone_length[zone_id] == 0.0:
            zone_length[zone_id] = round(length_val, 2)

    results = []
    for zone in range(1, 6):
        plant_identifier = f"{base_label}_plant_{zone}"
        results.append({"Plant ID": plant_identifier, "Length (px)": zone_length[zone]})

    if show_plots:
        plot_top_five_roots(refined_mask, base_label)

    return results

# === Optional Visualization ===
def plot_top_five_roots(refined_mask, identifier, area_min=1, top_row_limit=0.3):
    labeled = label(refined_mask > 0)
    img_h, img_w = refined_mask.shape
    root_entries = []

    for reg in regionprops(labeled):
        if reg.area < area_min:
            continue

        comp_mask = (labeled == reg.label).astype(np.uint8)
        skel_img = skeletonize(comp_mask)
        if np.count_nonzero(skel_img) == 0:
            continue

        skel_obj = Skeleton(skel_img)
        if skel_obj.n_paths == 0:
            continue

        if reg.bbox[0] > top_row_limit * img_h:
            continue

        adj = (skel_obj.graph != 0).astype(int)
        deg = adj.sum(axis=1).A.squeeze()
        endpoints = np.where(deg == 1)[0]
        if len(endpoints) < 2:
            continue

        coords = skel_obj.coordinates
        top_ep = min(endpoints, key=lambda idx: coords[idx, 0])
        distances, predecessors = dijkstra(csgraph=skel_obj.graph, directed=False,
                                           return_predecessors=True, indices=top_ep)
        far_ep = max(endpoints, key=lambda idx: distances[idx])
        length_val = distances[far_ep]

        min_r, min_c, max_r, max_c = reg.bbox
        bbox = (min_c, min_r, max_c - min_c, max_r - min_r)
        zone = determine_zone(bbox, img_w, img_h)

        root_entries.append({
            "label": reg.label,
            "zone": zone,
            "length": length_val,
            "skel": skel_obj,
            "start": top_ep,
            "end": far_ep,
            "pred": predecessors
        })

    top_roots = sorted(root_entries, key=lambda x: x["length"], reverse=True)[:5]
    top_roots.sort(key=lambda r: min(pt[1] for pt in r["skel"].coordinates))

    for idx, entry in enumerate(top_roots, start=1):
        path_coords = []
        current = entry["end"]
        while current != entry["start"]:
            path_coords.append(entry["skel"].coordinates[current])
            current = entry["pred"][current]
        path_coords.append(entry["skel"].coordinates[entry["start"]])
        path_coords = np.array(path_coords)

        plt.figure(figsize=(6, 6))
        plt.imshow(refined_mask, cmap='gray')
        plt.plot(path_coords[:, 1], path_coords[:, 0], "r-", linewidth=2)
        plt.title(f"Image: {identifier}, Root {idx} => Zone {entry['zone']}, Length={entry['length']:.2f}")
        plt.axis("off")
        plt.show()

        print(f"For image {identifier}, Region {entry['label']} => Zone {entry['zone']}, Length={entry['length']:.2f} px")
