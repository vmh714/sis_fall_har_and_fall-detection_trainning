import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Replace a specific cell in a Jupyter notebook.")
    parser.add_argument("--notebook", required=True, help="Path to the notebook (.ipynb)")
    parser.add_argument("--search-string", required=True, help="Unique string to identify the target cell (e.g., 'def build_model' or 'class KFoldDataManager')")
    parser.add_argument("--code-file", required=True, help="Path to the python file containing the new cell content")
    parser.add_argument("--output", help="Path to save the modified notebook. Overwrites original if not specified.")
    
    args = parser.parse_args()
    
    # Load notebook
    try:
        with open(args.notebook, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        sys.exit(1)
        
    # Load new code
    try:
        with open(args.code_file, 'r', encoding='utf-8') as f:
            new_code = f.read()
    except Exception as e:
        print(f"Error reading code file: {e}")
        sys.exit(1)
        
    # Format the new code into a list of lines with trailing newlines
    lines = new_code.split('\n')
    new_source = [line + '\n' for line in lines[:-1]]
    if lines:
        new_source.append(lines[-1]) # Last line doesn't need forced newline
        
    # Find target cell
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            if args.search_string in source_text:
                if target_idx != -1:
                    print(f"Warning: Multiple cells found containing '{args.search_string}'. Using the first one.")
                    break
                target_idx = i
                break
                
    if target_idx == -1:
        print(f"Error: Could not find any cell containing '{args.search_string}'")
        sys.exit(1)
        
    # Replace cell content
    nb['cells'][target_idx]['source'] = new_source
    
    # Save notebook
    out_path = args.output if args.output else args.notebook
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully replaced cell {target_idx} containing '{args.search_string}' in {out_path}")
    except Exception as e:
        print(f"Error saving notebook: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
