import matplotlib.pyplot as plt
import numpy as np

def generate_tracking_analysis(tracker, stats, output_path="output/tracking_analysis.png"):
    """Generate tracking analysis visualization"""

    import os
    os.makedirs("output", exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Player Tracking Analysis', fontsize=16, fontweight='bold')
    

    frames = range(1, len(tracker.players_per_frame) + 1)
    ax1.plot(frames, tracker.players_per_frame, 'b-', linewidth=2, alpha=0.7)
    ax1.fill_between(frames, tracker.players_per_frame, alpha=0.3)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Number of Players')
    ax1.set_title('Players Detected Per Frame')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    
    avg_players = np.mean(tracker.players_per_frame)

    ax1.axhline(y=avg_players, color='r', linestyle='--', 
                label=f'Average: {avg_players:.1f}')
    
    ax1.legend()
    
    # Player count distribution
    unique_counts, count_freq = np.unique(tracker.players_per_frame, return_counts=True)
    ax2.bar(unique_counts, count_freq, alpha=0.7, color='green')
    ax2.set_xlabel('Number of Players')
    ax2.set_ylabel('Frequency (Frames)')
    ax2.set_title('Distribution of Player Counts')
    ax2.grid(True, alpha=0.3)
    
    #  Tracking statistics summary
    ax3.axis('off')
    stats_text = f"""
    TRACKING STATISTICS
    
    Total Frames: {stats['total_frames']}
    Unique Players: {stats['unique_players']}
    Avg Players/Frame: {stats['avg_players_per_frame']:.2f}
    ID Consistency: {stats['id_consistency_rate']:.2%}
    Total Detections: {stats['total_detections']}
    
    Max Players in Frame: {max(tracker.players_per_frame) if tracker.players_per_frame else 0}
    Min Players in Frame: {min(tracker.players_per_frame) if tracker.players_per_frame else 0}
    """
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    
    # Show whenplayer ID was active
    player_timeline = {}
    
    
   
    for frame_idx, count in enumerate(tracker.players_per_frame):
     
        for player_id in range(1, min(stats['unique_players'] + 1, count + 1)):
            if player_id not in player_timeline:
                player_timeline[player_id] = []
            if len(player_timeline[player_id]) == 0 or frame_idx - player_timeline[player_id][-1] < 10:
                player_timeline[player_id].append(frame_idx)
    
    import matplotlib.colors as mcolors
    
    # Used a list of distinct colors 

    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = [mcolors.to_rgba(color_list[i % len(color_list)]) for i in range(min(10, stats['unique_players']))]
    for i, (player_id, frames_active) in enumerate(player_timeline.items()):
        if i >= 10:  
            break
        color = colors[i % len(colors)]
        y_pos = player_id
        
        # Group frames
        if frames_active:
            segments = []
            start = frames_active[0]
            prev = frames_active[0]
            
            for frame in frames_active[1:] + [frames_active[-1] + 2]: 
                if frame - prev > 1:  # Gap 
                    segments.append((start, prev))
                    start = frame
                prev = frame
            
            # Draw segments

            for seg_start, seg_end in segments:
                ax4.barh(y_pos, seg_end - seg_start + 1, left=seg_start, 
                        height=0.6, color=color, alpha=0.7)
    

    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Player ID')
    ax4.set_title('Player ID Timeline (Simplified)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.5, min(11, stats['unique_players'] + 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis visualization saved to: {output_path}")


def run_analysis():
    """Run the complete analysis"""
    
    from player_tracker import main
    
    print("Running player tracking analysis...")
    result = main()
    
    if result is None or not isinstance(result, tuple) or len(result) != 2:
        print("Error: Tracking failed")
        return None, None
        
    tracker, stats = result
    
    print("Generating analysis visualization...")
    generate_tracking_analysis(tracker, stats)
    
    return tracker, stats

if __name__ == "__main__":
    run_analysis()