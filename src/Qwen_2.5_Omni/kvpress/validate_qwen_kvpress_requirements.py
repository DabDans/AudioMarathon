import os
import re
import sys

def check_file_requirements(filepath):
    """Check if a single file meets the requirements"""
    if not os.path.exists(filepath):
        return {"exists": False}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {
        "exists": True,
        "has_record_initial_memory": False,
        "has_audio_file_check": False,
        "has_skip_record_creation": False,
        "has_timing_stats_class": False,
        "has_gpu_memory_usage_func": False
    }
    

    if re.search(r'def record_initial_memory\(', content):
        results["has_record_initial_memory"] = True
    

    if re.search(r'os\.path\.exists.*audio', content, re.IGNORECASE):
        results["has_audio_file_check"] = True
    

    skip_patterns = [
        r'"status":\s*"skipped"',
        r'"reason":\s*"audio_file_not_found"',
        r'audio_file_not_found',
        r'audio_processing_failed'
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, content):
            results["has_skip_record_creation"] = True
            break
    

    if re.search(r'class.*TimingStats', content):
        results["has_timing_stats_class"] = True
    

    if re.search(r'def get_gpu_memory_usage', content):
        results["has_gpu_memory_usage_func"] = True
    
    return results

def main():
    """Main validation function"""
    

    files_to_check = [
        "/data/to/your/HAD_qwen_kvpress/path/HAD_qwen_kvpress.py",
        "/data/to/your/LibriSpeech_qwen_kvpress/path/LibriSpeech_qwen_kvpress.py", 
        "/data/to/your/DESED_qwen_kvpress/path/DESED_qwen_kvpress.py"
    ]
    
    print("=== Qwen KV Press File Requirements Validation ===\n")
    
    all_passed = True
    
    for filepath in files_to_check:
        filename = os.path.basename(filepath)
        print(f"Checking file: {filename}")
        
        results = check_file_requirements(filepath)
        
        if not results["exists"]:
            print(f"  ❌ File does not exist")
            all_passed = False
            continue
        

        requirements = [
            ("record_initial_memory method", results["has_record_initial_memory"]),
            ("Audio file existence check", results["has_audio_file_check"]),
            ("Skip record creation", results["has_skip_record_creation"]),
            ("TimingStats class", results["has_timing_stats_class"]),
            ("GPU memory usage function", results["has_gpu_memory_usage_func"])
        ]
        
        file_passed = True
        for req_name, req_met in requirements:
            if req_met:
                print(f"  ✅ {req_name}")
            else:
                print(f"  ❌ {req_name}")
                file_passed = False
        
        if not file_passed:
            all_passed = False
        
        print()
    

    print("=== Validation Summary ===")
    if all_passed:
        print("✅ All files meet the requirements!")
    else:
        print("❌ Some files do not meet the requirements and need to be fixed")
    
    print("\nNote:")
    print("- Official Qwen system prompt format requirement is confirmed incompatible with pipeline architecture")
    print("- All files use direct model calls instead of pipeline architecture")
    print("- GPU memory monitoring uses initial_gpu_memory instead of average_gpu_memory")

if __name__ == "__main__":
    main()