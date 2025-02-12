import json
from pathlib import Path
import importlib.util

def convert_case_to_intermediate(floor_name: str, seg_pts: list, regions: list, wall_path: list) -> dict:
    """Convert case data to intermediate data format
    
    Args:
        floor_name: Name of the floor
        seg_pts: List of (x,y) coordinate tuples
        regions: List of ([indices], type) tuples
        wall_path: List of point indices for wall path
        
    Returns:
        Dictionary in intermediate_data.json format
    """
    return {
        'floor_name': floor_name,
        'seg_pts': seg_pts,
        'regions': regions,
        'wall_path': wall_path
    }

def convert_all_cases():
    """Convert all case files in cactus_data to intermediate format"""
    # Get the directory containing case files
    case_dir = Path(__file__).parent / 'cactus_data'
    output_dir = Path(__file__).parent.parent / 'output' / 'cases'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each case file
    for case_file in case_dir.glob('case*.py'):
        print(f"\n🔄 Processing {case_file.name}...")
        
        # Import the case module
        spec = importlib.util.spec_from_file_location(case_file.stem, str(case_file))
        case_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(case_module)
        
        # Convert to intermediate format
        intermediate_data = convert_case_to_intermediate(
            floor_name=case_file.stem,  # Use filename as floor name
            seg_pts=case_module.SEG_PTS,
            regions=case_module.CAC_REGIONS_FAKE,
            wall_path=case_module.WALL_PT_PATH
        )
        
        # Save to JSON file
        output_file = output_dir / f'{case_file.stem}_intermediate.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Converted {case_file.name} to {output_file.name}")

if __name__ == "__main__":
    print("\n🔷 Starting case data conversion...")
    convert_all_cases()
    print("\n✅ All cases converted successfully!") 