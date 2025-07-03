### Main Purpose
- Script compares the textual content ("prose") of controls and their parts from two OSCAL catalogs to identify semantically similar sections, helping users map or align controls between different security frameworks or versions.

### How It Works

1. **Flattening Controls and Parts:**
   - The function `flatten_controls_and_parts` recursively traverses the nested structure of an OSCAL catalog, extracting all controls and their "parts" (subsections), collecting their text ("prose") into a flat list. Each item in the list includes the text, its ID, and a reference to its parent control.

2. **Semantic Similarity Matching:**
   - The function `find_semantic_matches`:
     - Loads two OSCAL catalog JSON files (a "base" and a "merge" catalog).
     - Flattens both catalogs into lists of text snippets.
     - Uses the `sentence-transformers` library (specifically the `all-MiniLM-L6-v2` model) to encode these texts into vector embeddings.
     - Computes cosine similarity between all pairs of merge-catalog and base-catalog embeddings.
     - For each part in the merge catalog, it finds the top-k most similar parts in the base catalog, reporting those with a similarity score above a user-defined threshold.

3. **Output:**
   - For each control part in the merge catalog, the script prints:
     - The control and part IDs, and a snippet of the text.
     - The top-k most similar parts from the base catalog (if their similarity exceeds the threshold), including their IDs, similarity scores, and text snippets.
     - If no strong match is found, it notes this.

4. **Command-Line Interface:**
   - The script is run from the command line, requiring two file paths (base and merge catalogs), and optional parameters for similarity threshold and number of top matches to show.

### Use Cases
- Mapping controls between different security standards.
- Identifying overlaps or gaps between two sets of security requirements.
- Supporting compliance, audit, or migration tasks where semantic alignment of controls is needed.

### Key Technologies Used
- `sentence-transformers` for semantic text embeddings.
- `scikit-learn` for cosine similarity.
- `argparse` for command-line argument parsing.
- Standard Python libraries for JSON and file handling.
