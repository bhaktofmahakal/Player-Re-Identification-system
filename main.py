
"""Complete  Re-Identification Analysis Pipeline
Runs detection, tracking, visualization, and report generation"""

import os
import sys
import time
from player_tracker import main as run_tracking
from generate_analysis import generate_tracking_analysis
from generate_report import generate_report

def check_requirements():
    """Check if all required files exist"""

    required_files = [
        "data/models/yolov11_player_detection.pt",
        "data/videos/15sec_input_720p.mp4"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("SUCCESS: All required files found")
    return True

def main():
    """Run complete player re-identification pipeline"""

    print("Player Re-Identification System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\nERROR: Cannot proceed without required files")
        return False
    
    try:
        
        print("\nStep 1: Running Player Tracking...")
        print("-" * 30)
        start_time = time.time()
        
        tracker, stats = run_tracking()
        
        tracking_time = time.time() - start_time
        print(f"SUCCESS: Tracking completed in {tracking_time:.1f} seconds")
        
        # Validate results
        if tracker is None or stats is None:
            print("ERROR: Tracking failed - no results returned")
            return False
        
        #  Generate analysis 
        print("\nStep 2: Generating Analysis Visualization...")
        print("-" * 30)
        
        generate_tracking_analysis(tracker, stats)
        print("SUCCESS: Analysis visualization generated")
        
        # Generate report
        print("\nStep 3: Generating Report...")
        print("-" * 30)
        
        generate_report(stats)
        print("SUCCESS: Report generated")
        
        # Summary
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nOutput Files Generated in 'output/' folder:")
        print("   1. tracked_video.mp4 - Annotated video with player IDs")
        print("   2. tracking_analysis.png - Statistical analysis charts")
        print("   3. REPORT.md - Comprehensive system report")
        
        print(f"\nFinal Statistics:")
        print(f"   • Total Frames: {stats['total_frames']}")
        print(f"   • Unique Players: {stats['unique_players']}")
        print(f"   • Avg Players/Frame: {stats['avg_players_per_frame']:.2f}")
        print(f"   • ID Consistency: {stats['id_consistency_rate']:.1%}")
        print(f"   • Processing Time: {tracking_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)