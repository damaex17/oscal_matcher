#!/usr/bin/python3
import json
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

def flatten_controls_and_parts(catalog_data):
    """
    Recursively extracts all controls and their individual parts into a flat list.
    Each item in the list represents a piece of text to be compared.

    Args:
        catalog_data (dict): The loaded OSCAL catalog JSON data.

    Returns:
        list: A flat list of objects, each with parent control info and text.
    """
    flat_list = []

    def recurse_groups(groups):
        for group in groups:
            if 'controls' in group:
                for control in group['controls']:
                    if 'parts' in control and control['parts']:
                        for part in control['parts']:
                            def process_part(p, parent_control):
                                if 'prose' in p:
                                    flat_list.append({
                                        'id': p.get('id', parent_control.get('id')),
                                        'parent_control': parent_control,
                                        'text': p['prose']
                                    })
                                if 'parts' in p:
                                    for sub_part in p['parts']:
                                        process_part(sub_part, parent_control)
                            process_part(part, control)
                    elif 'prose' in control:
                        flat_list.append({
                            'id': control.get('id'),
                            'parent_control': control,
                            'text': control['prose']
                        })
            if 'groups' in group:
                recurse_groups(group['groups'])

    if 'catalog' in catalog_data and 'groups' in catalog_data['catalog']:
        recurse_groups(catalog_data['catalog']['groups'])

    return flat_list

def find_semantic_matches(base_catalog_path, merge_catalog_path, threshold=0.65, top_k=3):
    """
    Finds and prints semantic matches between two OSCAL catalogs based on control parts.

    Args:
        base_catalog_path (str): Path to the base OSCAL catalog file.
        merge_catalog_path (str): Path to the OSCAL catalog to merge.
        threshold (float): The minimum similarity score to consider a match.
        top_k (int): The number of top matches to display for each control part.
    """
    print("Loading catalogs...")
    try:
        with open(base_catalog_path, 'r', encoding='utf-8') as f:
            base_catalog = json.load(f)
        with open(merge_catalog_path, 'r', encoding='utf-8') as f:
            merge_catalog = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file: {e}")
        return

    print("Flattening controls and parts from both catalogs...")
    base_flat_list = flatten_controls_and_parts(base_catalog)
    merge_flat_list = flatten_controls_and_parts(merge_catalog)

    if not base_flat_list or not merge_flat_list:
        print("Error: No controls or parts with prose found in one or both catalogs.")
        return

    base_texts = [item['text'] for item in base_flat_list]
    merge_texts = [item['text'] for item in merge_flat_list]

    print("Loading sentence transformer model... (This may take a moment on first run)")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding control and part descriptions into vectors...")
    base_embeddings = model.encode(base_texts, convert_to_tensor=True)
    merge_embeddings = model.encode(merge_texts, convert_to_tensor=True)

    print("Calculating similarities...")
    similarities = cosine_similarity(merge_embeddings.cpu(), base_embeddings.cpu())

    print("\n--- Potential Control Matches Report ---")
    merge_file_name = os.path.basename(merge_catalog_path)
    base_file_name = os.path.basename(base_catalog_path)
    print(f"Finding top {top_k} matches for each control part in '{merge_file_name}' with a similarity score > {threshold}\n")

    for i, merge_item in enumerate(merge_flat_list):
        sim_scores = similarities[i]

        # Get the indices of the top_k scores
        top_indices = torch.topk(torch.tensor(sim_scores), k=top_k).indices.tolist()

        merge_parent_control = merge_item['parent_control']
        print("--------------------------------------------------")
        print(f"[*] Source Control Part from '{merge_file_name}':")
        print(f"    Control ID: {merge_parent_control.get('id', 'N/A')}")
        print(f"    Part ID: {merge_item.get('id', 'N/A')}")
        print(f"    Text: \"{merge_item['text'][:200]}...\"")
        print("--------------------------------------------------")

        found_match = False
        for index in top_indices:
            score = sim_scores[index]
            if score >= threshold:
                if not found_match:
                    print(f"  -> Potential Matches in '{base_file_name}':")
                    found_match = True

                base_item = base_flat_list[index]
                base_parent_control = base_item['parent_control']
                print(f"    - Match Score: {score:.2f}")
                print(f"      Control ID: {base_parent_control.get('id', 'N/A')}")
                print(f"      Part ID: {base_item.get('id', 'N/A')}")
                print(f"      Text: \"{base_item['text'][:200]}...\"\n")

        if not found_match:
            print("  -> No strong matches found in the base catalog.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find semantic matches between control parts of two OSCAL catalogs.")
    parser.add_argument("base_catalog", help="Path to the base OSCAL catalog JSON file.")
    parser.add_argument("merge_catalog", help="Path to the OSCAL catalog to find matches for.")
    parser.add_argument("--threshold", type=float, default=0.65, help="Minimum similarity score to consider a match (0.0 to 1.0).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top potential matches to show.")

    args = parser.parse_args()

    find_semantic_matches(args.base_catalog, args.merge_catalog, args.threshold, args.top_k)
