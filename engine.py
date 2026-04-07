import os
import cv2
import numpy as np
import pandas as pd
import torch
import shutil
import streamlit as st
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import exifread
import insightface
from insightface.app import FaceAnalysis

class BatchAnalyzer:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load InsightFace (ArcFace + RetinaFace)
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0 if self.device.type == 'cuda' else -1, det_size=(640, 640))

        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        self.scenery_labels = {
            'Nature': ['nature', 'mountain', 'beach', 'forest', 'waterfall', 'landscape'],
            'City': ['city street', 'buildings', 'urban', 'skyline'],
            'Food': ['food', 'plate of food', 'restaurant', 'drink'],
            'Monument': ['monument', 'landmark', 'temple', 'statue', 'cathedral', 'historic building']
        }
        self.prompts = []
        self.prompt_to_cat = []
        for cat, ps in self.scenery_labels.items():
            for p in ps:
                self.prompts.append(f'a photo of {p}')
                self.prompt_to_cat.append(cat)

    def get_timestamp(self, path):
        try:
            with open(path, 'rb') as f:
                tags = exifread.process_file(f, stop_tag='DateTimeOriginal', details=False)
                ts = tags.get('EXIF DateTimeOriginal')
                return str(ts) if ts else None
        except:
            return None

    def quality_check(self, path):
        try:
            img = cv2.imread(path)
            if img is None: return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            if blur < 60 or brightness < 20:
                return None
            return {'path': path, 'blur': blur, 'brightness': brightness, 'shape': img.shape}
        except:
            return None

    def get_body_embedding(self, bgr, face_bbox):
        """Extracts clothing/body embedding using CLIP."""
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = face_bbox.astype(int)
        
        # Extend box downwards to capture torso/clothes
        face_h = y2 - y1
        body_y1 = y1
        body_y2 = min(h, y2 + (face_h * 4)) # Capture roughly 4x face height down
        body_x1 = max(0, x1 - face_h)
        body_x2 = min(w, x2 + face_h)
        
        body_crop = bgr[body_y1:body_y2, body_x1:body_x2]
        if body_crop.size == 0:
            return None
            
        pil_body = Image.fromarray(cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB))
        inputs = self.clip_processor(images=pil_body, return_tensors='pt').to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        return emb.cpu().numpy().flatten()

    def face_quality_check(self, bgr, faces):
        if not faces:
            return True, []

        face_statuses = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w = bgr.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            face_crop = bgr[y1:y2, x1:x2]
            if face_crop.size == 0:
                face_statuses.append(False)
                continue

            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Simple blur check
            is_good = face_blur > 40
            face_statuses.append(is_good)
        
        return all(face_statuses), face_statuses

    def get_aesthetic_score_and_embedding(self, pil_images):
        prompts = ['a professional, high-quality, beautiful photo', 'a blurry, low-quality, bad amateur photo'] 
        inputs = self.clip_processor(text=prompts, images=pil_images, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            embeddings = self.clip_model.get_image_features(pixel_values=inputs['pixel_values'])
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

        return probs[:, 0], embeddings.cpu().numpy()

    def process_batch(self, paths, batch_size=8):
        with ThreadPoolExecutor() as executor:
            quality_results = list(executor.map(self.quality_check, paths))
            timestamps = list(executor.map(self.get_timestamp, paths))

        valid_data = []
        for p, q, ts in zip(paths, quality_results, timestamps):
            if q:
                q['timestamp'] = ts
                valid_data.append(q)

        if not valid_data: return pd.DataFrame()

        final_valid = []
        for data in valid_data:
            try:
                bgr = cv2.imread(data['path'])
                if bgr is None: continue
                
                faces = self.face_app.get(bgr)
                is_face_ok, _ = self.face_quality_check(bgr, faces)
                if not is_face_ok: continue

                data['num_faces'] = len(faces)
                data['faces'] = faces
                
                # IMPORTANT: Initialize body_embedding for all valid rows to prevent KeyError
                if len(faces) == 1:
                    data['body_embedding'] = self.get_body_embedding(bgr, faces[0].bbox)
                else:
                    data['body_embedding'] = None
                    
                final_valid.append(data)
            except Exception as e:
                print(f"Error processing {data['path']}: {e}")
                continue

        if not final_valid: return pd.DataFrame()

        for j in range(0, len(final_valid), batch_size):
            sub_batch = final_valid[j : j + batch_size]
            pil_imgs = [Image.open(d['path']).convert('RGB') for d in sub_batch]
            scores, v_embs = self.get_aesthetic_score_and_embedding(pil_imgs)
            for k, (score, emb) in enumerate(zip(scores, v_embs)):
                sub_batch[k]['aesthetic_score'] = score
                sub_batch[k]['visual_embedding'] = emb

        final_valid = [d for d in final_valid if d['aesthetic_score'] > 0.4]

        for d in final_valid:
            if d['num_faces'] == 0:
                pil = Image.open(d['path']).convert('RGB')
                inputs = self.clip_processor(text=self.prompts, images=pil, return_tensors='pt', padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.clip_model(**inputs).logits_per_image
                    best_idx = logits.softmax(dim=1).cpu().numpy().argmax()
                d['category'] = self.prompt_to_cat[best_idx]
                d['person'] = None
            elif d['num_faces'] == 1:
                d['category'] = 'Solo'
                d['person'] = d['faces'][0].normed_embedding.tolist()
            else:
                d['category'] = 'Group'
                d['person'] = 'GROUP'

        # Ensure all required columns exist even if no faces were found in this specific run
        df = pd.DataFrame(final_valid)
        if not df.empty and 'body_embedding' not in df.columns:
            df['body_embedding'] = None
        return df

def cluster_events(df, eps_minutes=60):
    if df.empty or 'timestamp' not in df.columns: return df
    df['dt'] = pd.to_datetime(df['timestamp'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    df['dt'] = df['dt'].ffill().bfill()
    if df['dt'].isna().all():
        df['event'] = 'Event_0'
        return df
    df['unix'] = df['dt'].view('int64') // 10**9
    X = df[['unix']].values
    clustering = DBSCAN(eps=eps_minutes * 60, min_samples=1).fit(X)
    df['event'] = [f'Event_{l}' for l in clustering.labels_]
    return df

def cluster_people(df):
    """Refined clustering using both Face and Body/Clothing logic."""
    if df.empty: return df
    
    # Safety check for column existence
    if 'body_embedding' not in df.columns:
        df['body_embedding'] = None

    mask = (df['category'] == 'Solo') & df['person'].apply(lambda x: isinstance(x, list))
    if mask.sum() < 1:
        df.loc[df['category'] == 'Solo', 'person'] = 'Unknown_Person'
        return df
    
    # 1. Global Face Vectors
    face_vectors = np.array(df.loc[mask, 'person'].tolist())
    face_vectors = normalize(face_vectors)
    
    # 2. Body Vectors (for local event refinement)
    body_data = df.loc[mask, 'body_embedding'].values
    event_ids = df.loc[mask, 'event'].values
    
    # Custom similarity matrix for DBSCAN
    n = len(face_vectors)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Base Face Similarity (Cosine)
            face_sim = np.dot(face_vectors[i], face_vectors[j])
            
            # Contextual weight
            if event_ids[i] == event_ids[j] and body_data[i] is not None and body_data[j] is not None:
                # Same event: combine Face and Clothing
                body_sim = cosine_similarity(body_data[i].reshape(1, -1), body_data[j].reshape(1, -1))[0][0]
                # High clothing similarity can "pull" borderline faces together
                combined_sim = (0.6 * face_sim) + (0.4 * body_sim)
            else:
                # Different event: rely primarily on Face (as clothes likely changed)
                combined_sim = face_sim
            
            # Convert similarity to distance for DBSCAN (1 - sim)
            dist = max(0, 1.0 - combined_sim)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # eps=0.35 (distance) corresponds to sim=0.65
    clustering = DBSCAN(eps=0.35, min_samples=1, metric='precomputed').fit(dist_matrix)
    
    labels = [f'Person_{label+1}' if label != -1 else 'Unknown_Person' for label in clustering.labels_]
    df.loc[mask, 'person'] = labels
    df.loc[(df['category'] == 'Solo') & ~mask, 'person'] = 'Unknown_Person'
    return df

def filter_redundant(df, similarity_threshold=0.92):
    if df.empty: return df
    filtered_indices = []
    for event in df['event'].unique():
        event_df = df[df['event'] == event].sort_values(by='aesthetic_score', ascending=False)
        if event_df.empty: continue
        event_indices = event_df.index.tolist()
        keep = [event_indices[0]]
        embeddings = np.stack(event_df['visual_embedding'].values)
        for i in range(1, len(event_indices)):
            curr_idx = event_indices[i]
            curr_emb = embeddings[i].reshape(1, -1)
            is_redundant = False
            for kept_idx in keep:
                kept_pos = event_indices.index(kept_idx)
                kept_emb = embeddings[kept_pos].reshape(1, -1)
                sim = cosine_similarity(curr_emb, kept_emb)[0][0]
                if sim > similarity_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                keep.append(curr_idx)
        filtered_indices.extend(keep)
    return df.loc[filtered_indices].sort_index()

def export_organized(df, export_dir):
    for _, row in df.iterrows():
        event = row.get('event', 'Unsorted')
        cat = row['category']
        if cat == 'Solo':
            person = row['person'] if isinstance(row['person'], str) else 'Unidentified'
            subfolder = os.path.join('By_Event', event, 'Solo_Pics', person)
        elif cat == 'Group':
            subfolder = os.path.join('By_Event', event, 'Group_Photos')
        else:
            subfolder = os.path.join('By_Event', event, 'Scenery', cat)
        dest_path = os.path.join(export_dir, subfolder)
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy2(row['path'], os.path.join(dest_path, os.path.basename(row['path'])))

    solo_df = df[df['category'] == 'Solo']
    for person_id in solo_df['person'].unique():
        p_folder = person_id if isinstance(person_id, str) else 'Unidentified'
        person_path = os.path.join(export_dir, 'People', p_folder)
        os.makedirs(person_path, exist_ok=True)
        subset = solo_df[solo_df['person'] == person_id]
        for _, row in subset.iterrows():
            shutil.copy2(row['path'], os.path.join(person_path, os.path.basename(row['path'])))

    highlights_path = os.path.join(export_dir, 'Highlights')
    os.makedirs(highlights_path, exist_ok=True)
    for event_id in df['event'].unique():
        event_group = df[df['event'] == event_id]
        top_photos = event_group.nlargest(3, 'aesthetic_score')
        for _, row in top_photos.iterrows():
            prefix = f'{event_id}_'
            shutil.copy2(row['path'], os.path.join(highlights_path, prefix + os.path.basename(row['path'])))
