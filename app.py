import streamlit as st
import os
import pandas as pd
from engine import BatchAnalyzer, cluster_events, cluster_people, export_organized, filter_redundant
from drive_utils import DriveHandler
import time
import tkinter as tk
from tkinter import filedialog

# --- CONFIG ---
st.set_page_config(page_title="Efficient Trip Organizer", layout="wide")
TEMP_DRIVE_DIR = "temp_drive_files"

def select_folder():
    """Opens a native Windows folder selection dialog."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.askdirectory(master=root)
    root.destroy()
    return path

@st.cache_resource
def get_analyzer():
    return BatchAnalyzer()

def main():
    st.title("📸 Efficient Trip Memory Organizer")
    st.markdown("Organize your photos by events, people, and scenery using AI-powered batch processing.")

    # --- SIDEBAR: INPUT CONFIG ---
    st.sidebar.header("Input Settings")
    input_mode = st.sidebar.radio("Input Source", ["Local Folder", "Google Drive"])
    
    folder_path = ""
    drive_folder_id = ""
    
    if input_mode == "Local Folder":
        folder_path = st.text_input("Local Folder Path", placeholder="e.g., C:/Users/Me/Pictures/Trip")
    else:
        drive_folder_id = st.text_input("Google Drive Folder ID")
        st.info("The ID is the long string at the end of the folder's URL.")

    # --- SETTINGS ---
    st.sidebar.header("Sorting Settings")
    eps_min = st.sidebar.slider("Event Gap (Minutes)", 15, 240, 60, help="Time gap to consider a new event.")
    redundancy_threshold = st.sidebar.slider("Redundancy Sensitivity", 0.80, 0.99, 0.92, 0.01, help="Higher = keeps more similar photos. Lower = stricter filtering.")
    
    if st.button("🚀 Start Organizing", type="primary"):
        analyzer = get_analyzer()
        files = []

        # 1. Gather Files
        with st.status("Gathering files...", expanded=True) as status:
            if input_mode == "Local Folder":
                if not os.path.isdir(folder_path):
                    st.error("Invalid local path!")
                    st.stop()
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            else:
                if not drive_folder_id:
                    st.error("Please enter a Drive Folder ID!")
                    st.stop()
                try:
                    handler = DriveHandler("credentials.json")
                    st.write("Downloading files from Drive...")
                    files = handler.download_folder(drive_folder_id, TEMP_DRIVE_DIR)
                except Exception as e:
                    st.error(f"Drive Error: {e}")
                    st.stop()
            
            st.write(f"Found {len(files)} images.")
            
            # 2. Batch Processing
            st.write("Analyzing images (InsightFace + CLIP)...")
            start_time = time.time()
            df = analyzer.process_batch(files)
            duration = time.time() - start_time
            
            if df.empty:
                st.warning("No valid images found after quality check.")
                st.stop()
            
            st.write(f"Processed {len(df)} images in {duration:.2f} seconds.")

            # 3. Clustering & Filtering
            st.write("Filtering redundant photos...")
            df = cluster_events(df, eps_minutes=eps_min)
            df = filter_redundant(df, similarity_threshold=redundancy_threshold)
            
            st.write("Identifying people...")
            df = cluster_people(df)
            
            # Save to session state
            st.session_state['df'] = df
            st.session_state['analysis_done'] = True
            
            status.update(label="Analysis Complete!", state="complete")

    # --- DISPLAY RESULTS (Outside button block) ---
    if st.session_state.get('analysis_done'):
        df = st.session_state['df']
        st.success(f"Analysis complete! Final count: {len(df)} images (after removing duplicates).")

        # --- PREVIEW ---
        tab1, tab2, tab3 = st.tabs(["📊 Stats", "🖼️ Preview", "👤 Name People"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Events", len(df['event'].unique()))
            # Ensure we only count string labels for people
            solo_people = df[(df['category'] == 'Solo') & (df['person'].apply(lambda x: isinstance(x, str)))]
            unique_people = sorted([p for p in solo_people['person'].unique() if p != 'Unidentified'])
            col2.metric("People Identified", len(unique_people))
            col3.metric("Group Photos", len(df[df['category'] == 'Group']))
            
            st.info("💡 Photos will be organized into 'People' and 'By Event' structures.")
            st.dataframe(df[['path', 'category', 'event', 'person', 'aesthetic_score']].head(50))

        with tab2:
            events = sorted(df['event'].unique())
            selected_event = st.selectbox("Select Event to Preview", events)
            event_df = df[df['event'] == selected_event]
            
            for cat in sorted(event_df['category'].unique()):
                st.subheader(f"{cat}")
                subset = event_df[event_df['category'] == cat]
                cols = st.columns(5)
                for i, (_, row) in enumerate(subset.iterrows()):
                    with cols[i % 5]:
                        st.image(row['path'], use_container_width=True)
                        if cat == "Solo":
                            st.caption(f"ID: {row['person']}")
                        st.caption(f"Score: {row['aesthetic_score']:.2f}")

        with tab3:
            st.subheader("Map Person IDs to Names")
            st.write("Enter names for the identified people to rename their export folders.")
            
            # Create a mapping dict in session state if not exists
            if 'name_map' not in st.session_state:
                st.session_state['name_map'] = {p: p for p in unique_people}
            
            mapping_cols = st.columns(3)
            for idx, p_id in enumerate(unique_people):
                with mapping_cols[idx % 3]:
                    # Show a sample image of the person
                    sample_img = solo_people[solo_people['person'] == p_id].iloc[0]['path']
                    st.image(sample_img, width=100)
                    st.session_state['name_map'][p_id] = st.text_input(f"Name for {p_id}", value=st.session_state['name_map'].get(p_id, p_id), key=f"input_{p_id}")

        # --- EXPORT ---
        st.divider()
        st.subheader("📁 Final Export")
        st.write("Click below to choose a location on your computer and save the organized photos.")
        
        if st.button("🚀 Select Folder & Export Now", type="primary"):
            selected_path = select_folder()
            
            if not selected_path:
                st.warning("Export cancelled. No folder selected.")
            else:
                with st.spinner(f"Exporting to {selected_path}..."):
                    # Apply names to the dataframe
                    final_df = df.copy()
                    if 'name_map' in st.session_state:
                        final_df['person'] = final_df['person'].replace(st.session_state['name_map'])
                    
                    export_organized(final_df, selected_path)
                    st.success(f"✅ Success! Photos exported to: `{selected_path}`")
                    st.balloons()
                    # Open the folder automatically (Windows only)
                    if os.name == 'nt':
                        os.startfile(selected_path)

if __name__ == "__main__":
    main()
