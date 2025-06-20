import cv2
import numpy as np
import time
import json
import threading
import argparse
from datetime import datetime, timezone
from collections import deque
import pytz
from ultralytics import YOLO
from ai_playground import run_ocr
from process_JSON import process_json, process_task_timestamps, initialize_firebase, get_all_tasks_map_sync
from firebase_admin import firestore

# Configuration
MODEL_PATH = "my_model.pt"  # You can change this to your custom model path
CONFIDENCE_THRESHOLD = 0.3
WEBCAM_INDEX = 0  # Usually 0 for default webcam on Mac

# Turbo Mode Storage
turbo_stored_json = None
turbo_stored_order = None
turbo_daily_meta = None
turbo_daily_id = None

# AUTO MODE DETECTION - NEW VARIABLES
auto_mode_enabled = True  # Enable/disable automatic mode detection
previous_symbol_count = 0  # Track previous symbol count
current_symbol_count = 0   # Track current symbol count
symbol_count_history = deque(maxlen=15)  # Track symbol count over time for stability
stability_threshold = 10   # Frames needed for stability
auto_turbo_cooldown = 0   # Cooldown to prevent rapid-fire processing
auto_ocr_cooldown = 0     # Cooldown for OCR processing

# FULL PAGE VIEW DETECTION - NEW VARIABLES
page_view_detector = {
    'symbol_positions': [],
    'position_history': deque(maxlen=20),
    'stable_frames': 0,
    'required_stable_frames': 15,
    'position_threshold': 30,  # pixels
    'waiting_for_full_view': False,
    'max_wait_time': 300,  # 5 seconds at 60fps
    'wait_counter': 0
}

# INITIAL SCAN DETECTION - NEW VARIABLES
initial_scan_detector = {
    'has_scanned_once': False,  # Track if we've done the initial scan
    'symbol_count_history': deque(maxlen=25),  # Longer history for initial detection
    'max_symbol_count_seen': 0,  # Track the maximum symbol count seen
    'stable_count_frames': 0,  # Frames where count has been stable
    'required_stable_frames': 20,  # More frames required for initial scan
    'min_symbol_threshold': 3,  # Minimum symbols to consider a valid page
    'growth_stopped_frames': 0,  # Frames since symbol count stopped growing
    'growth_stop_threshold': 15,  # Frames to wait after growth stops
    'edge_margin': 50,  # Pixels from edge to consider "too close to edge"
    'waiting_for_initial_scan': False,
    'initial_scan_cooldown': 0,  # Cooldown to prevent multiple attempts
}

# Set bounding box colors for each label
label_colors = {
    "COMPLETED": (0, 255, 0),      # Green - success/done
    "IN_PROGRESS": (255, 165, 0),  # Orange - work in progress
    "MEETING": (0, 165, 255),      # Blue - meetings/scheduled
    "NOT_STARTED": (0, 0, 255),    # Red - not started/urgent
    "TEXT_AREA": (128, 0, 128)     # Purple - text areas
}

def apply_image_transformations(frame, orientation_mode="phone"):
    """
    Apply image transformations based on orientation mode
    
    Args:
        frame: Input frame from webcam
        orientation_mode: "phone" or "landscape"
            - "phone": Apply 90° clockwise rotation (current behavior)
            - "landscape": Apply transformations to make image vertical
    """
    transformed_frame = frame.copy()
    
    if orientation_mode == "phone":
        # Apply 90 degree clockwise rotation (current behavior)
        transformed_frame = cv2.rotate(transformed_frame, cv2.ROTATE_90_CLOCKWISE)
    elif orientation_mode == "landscape":
        # No rotation applied - original webcam orientation
        # Previous 90° counter-clockwise + additional 90° clockwise = 0° total rotation
        pass  # No transformation needed
    
    return transformed_frame

def perform_ocr_async(frame_copy):
    """Send frame for OCR and return the extracted text - runs in background thread"""
    try:
        print("🔍 [Background] Starting OCR processing...")
        
        result = run_ocr(frame_copy)
        
        if result:
            print("✅ [Background] OCR completed successfully!")
            print("OCR Result:")
            print("-" * 50)
            print(result)
            print("-" * 50)
        else:
            print("❌ [Background] No OCR result received")
    except Exception as e:
        print(f"❌ [Background] Error in OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()

def process_json_from_ocr_async(frame_copy):
    """Get JSON from OCR and process it through process_JSON - runs in background thread"""
    try:
        print("🔄 [Background] Starting JSON processing from OCR...")
        
        # Step 1: Get OCR result
        print("📸 [Background] Capturing frame and running OCR...")
        ocr_result = run_ocr(frame_copy)
        
        if not ocr_result:
            print("❌ [Background] No OCR result received")
            return
        
        print("✅ [Background] OCR completed successfully")
        print("[Background] Raw OCR Result:")
        print("-" * 50)
        print(ocr_result)
        print("-" * 50)
        
        # Step 2: Try to parse JSON from OCR result
        try:
            # Try to find JSON in the OCR result
            # The OCR result might contain JSON directly or be wrapped in text/markdown
            json_data = None
            
            # First try to parse the entire result as JSON
            try:
                json_data = json.loads(ocr_result)
                print("✅ [Background] Successfully parsed OCR result as direct JSON")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code blocks
                # Look for ```json ... ``` or ``` ... ``` blocks
                lines = ocr_result.split('\n')
                json_content = []
                in_code_block = False
                
                for line in lines:
                    line_stripped = line.strip()
                    
                    # Check for start of code block
                    if line_stripped.startswith('```'):
                        if line_stripped == '```json' or line_stripped == '```':
                            in_code_block = True
                            continue
                        elif in_code_block:
                            # End of code block
                            break
                    
                    # Collect JSON content inside code block
                    if in_code_block:
                        json_content.append(line)
                
                # Try to parse the extracted content
                if json_content:
                    json_string = '\n'.join(json_content)
                    try:
                        json_data = json.loads(json_string)
                        print("✅ [Background] Successfully extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        pass
                
                # If markdown extraction failed, try to find single-line JSON
                if not json_data:
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                json_data = json.loads(line)
                                print("✅ [Background] Successfully extracted JSON from single line")
                                break
                            except json.JSONDecodeError:
                                continue
            
            if not json_data:
                print("❌ [Background] Could not parse JSON from OCR result")
                print("💡 [Background] OCR result should contain valid JSON with 'tasks' key")
                return
            
            print("📋 [Background] Parsed JSON data:")
            print(json.dumps(json_data, indent=2))
            
        except Exception as e:
            print(f"❌ [Background] Error parsing JSON from OCR: {str(e)}")
            return
        
        # Step 3: Prepare metadata for process_json
        local_tz = pytz.timezone('Asia/Bangkok')  # UTC+7
        now = datetime.now(local_tz)
        daily_id = now.strftime("%Y-%m-%d")
        
        daily_meta = {
            "date": daily_id,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "cardScannedAt": now.isoformat()
        }
        
        print(f"📅 [Background] Processing for date: {daily_id}")
        print("🔧 [Background] Daily metadata:")
        print(json.dumps(daily_meta, indent=2))
        
        # Step 4: Call process_json
        print("🚀 [Background] Calling process_json...")
        result = process_json(json_data, daily_meta, daily_id)
        
        # Step 5: Log results
        if result.success:
            print("🎉 [Background] SUCCESS!")
            print(f"[Background] {result.message}")
            if result.data:
                print("📊 [Background] Processed data structure:")
                print(f"   - Tasks processed: {len(result.data.get('tasks', []))}")
                print(f"   - Daily metadata: {result.data.get('daily', {}).keys()}")
        else:
            print("💥 [Background] FAILED!")
            print(f"[Background] {result.message}")
            
    except Exception as e:
        print(f"❌ [Background] Unexpected error in process_json_from_ocr: {str(e)}")
        import traceback
        traceback.print_exc()

def process_json_from_ocr_async_with_turbo_storage(frame_copy, detections, labels):
    """Get JSON from OCR and process it through process_JSON - also stores for turbo mode"""
    global turbo_stored_json, turbo_stored_order, turbo_daily_meta, turbo_daily_id
    
    try:
        print("🔄 [Background] Starting JSON processing from OCR with turbo storage...")
        
        # Step 1: Get OCR result
        print("📸 [Background] Capturing frame and running OCR...")
        ocr_result = run_ocr(frame_copy)
        
        if not ocr_result:
            print("❌ [Background] No OCR result received")
            return
        
        print("✅ [Background] OCR completed successfully")
        print("[Background] Raw OCR Result:")
        print("-" * 50)
        print(ocr_result)
        print("-" * 50)
        
        # Step 2: Try to parse JSON from OCR result
        try:
            # Try to find JSON in the OCR result
            # The OCR result might contain JSON directly or be wrapped in text/markdown
            json_data = None
            
            # First try to parse the entire result as JSON
            try:
                json_data = json.loads(ocr_result)
                print("✅ [Background] Successfully parsed OCR result as direct JSON")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code blocks
                # Look for ```json ... ``` or ``` ... ``` blocks
                lines = ocr_result.split('\n')
                json_content = []
                in_code_block = False
                
                for line in lines:
                    line_stripped = line.strip()
                    
                    # Check for start of code block
                    if line_stripped.startswith('```'):
                        if line_stripped == '```json' or line_stripped == '```':
                            in_code_block = True
                            continue
                        elif in_code_block:
                            # End of code block
                            break
                    
                    # Collect JSON content inside code block
                    if in_code_block:
                        json_content.append(line)
                
                # Try to parse the extracted content
                if json_content:
                    json_string = '\n'.join(json_content)
                    try:
                        json_data = json.loads(json_string)
                        print("✅ [Background] Successfully extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        pass
                
                # If markdown extraction failed, try to find single-line JSON
                if not json_data:
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                json_data = json.loads(line)
                                print("✅ [Background] Successfully extracted JSON from single line")
                                break
                            except json.JSONDecodeError:
                                continue
            
            if not json_data:
                print("❌ [Background] Could not parse JSON from OCR result")
                print("💡 [Background] OCR result should contain valid JSON with 'tasks' key")
                return
            
            print("📋 [Background] Parsed JSON data:")
            print(json.dumps(json_data, indent=2))
            
        except Exception as e:
            print(f"❌ [Background] Error parsing JSON from OCR: {str(e)}")
            return
        
        # Step 3: Prepare metadata for process_json
        local_tz = pytz.timezone('Asia/Bangkok')  # UTC+7
        now = datetime.now(local_tz)
        daily_id = now.strftime("%Y-%m-%d")
        
        daily_meta = {
            "date": daily_id,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "cardScannedAt": now.isoformat()
        }
        
        print(f"📅 [Background] Processing for date: {daily_id}")
        print("🔧 [Background] Daily metadata:")
        print(json.dumps(daily_meta, indent=2))
        
        # Step 4: Call process_json
        print("🚀 [Background] Calling process_json...")
        result = process_json(json_data, daily_meta, daily_id)
        
        # Step 5: Log results
        if result.success:
            print("🎉 [Background] SUCCESS!")
            print(f"[Background] {result.message}")
            if result.data:
                print("📊 [Background] Processed data structure:")
                print(f"   - Tasks processed: {len(result.data.get('tasks', []))}")
                print(f"   - Daily metadata: {result.data.get('daily', {}).keys()}")
            
            # Step 6: Store for turbo mode
            print("💾 [Turbo] Storing data for turbo mode...")
            turbo_stored_json = json_data
            turbo_stored_order = get_yolo_order_top_to_bottom(detections, labels)
            turbo_daily_meta = daily_meta
            turbo_daily_id = daily_id
            
            print(f"✅ [Turbo] Stored JSON with {len(json_data.get('tasks', []))} tasks")
            print(f"✅ [Turbo] Stored YOLO order: {turbo_stored_order}")
            print("🚀 [Turbo] Ready for turbo mode! Press 'w' to use fast processing.")
            
        else:
            print("💥 [Background] FAILED!")
            print(f"[Background] {result.message}")
            
    except Exception as e:
        print(f"❌ [Background] Unexpected error in process_json_from_ocr: {str(e)}")
        import traceback
        traceback.print_exc()

def get_yolo_order_top_to_bottom(detections, labels):
    """Extract YOLO detection order sorted by Y-coordinate (top to bottom)"""
    if detections is None:
        return []
    
    detection_data = []
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        
        # Get class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        
        # Get confidence score
        conf = detections[i].conf.item()
        
        # Only include detections above threshold and exclude TEXT_AREA
        if conf > CONFIDENCE_THRESHOLD and classname != "TEXT_AREA":
            detection_data.append({
                'class': classname,
                'y_center': (ymin + ymax) / 2,
                'confidence': conf
            })
    
    # Sort by Y coordinate (top to bottom)
    detection_data.sort(key=lambda x: x['y_center'])
    
    # Return just the class names in order
    return [item['class'] for item in detection_data]

def update_json_with_new_order(stored_json, new_order, existing_tasks_map=None):
    """Update the stored JSON with new status order and proper timestamp handling"""
    if not stored_json or 'tasks' not in stored_json:
        return None
    
    updated_json = json.loads(json.dumps(stored_json))  # Deep copy
    tasks = updated_json.get('tasks', [])
    
    # Get current time for timestamp processing
    local_tz = pytz.timezone('Asia/Bangkok')  # UTC+7
    current_time = datetime.now(local_tz)
    
    # Update task statuses and timestamps based on new order
    for i, new_status in enumerate(new_order):
        if i < len(tasks):
            # Get previous task data (if we have existing tasks map)
            prev_task = None
            if existing_tasks_map:
                task_name = tasks[i].get('name', '')
                prev_task = existing_tasks_map.get(task_name)
            
            # Update status
            old_status = tasks[i].get('status', '')
            tasks[i]['status'] = new_status
            
            # Create a temporary task object for timestamp processing
            temp_task = {
                'name': tasks[i].get('name', ''),
                'status': new_status,
                'startedAt': tasks[i].get('startedAt'),
                'completedAt': tasks[i].get('completedAt'),
                'plannedAt': tasks[i].get('plannedAt')
            }
            
            # Process timestamps using the same logic as process_JSON
            planned_at, started_at, completed_at = process_task_timestamps(
                temp_task, 
                prev_task, 
                current_time
            )
            
            # Update the task with proper timestamps
            tasks[i]['plannedAt'] = planned_at.isoformat() if planned_at else None
            tasks[i]['startedAt'] = started_at.isoformat() if started_at else None
            tasks[i]['completedAt'] = completed_at.isoformat() if completed_at else None
            
            print(f"📝 [Turbo] Updated task {i}: '{tasks[i].get('name', 'Unknown')}' {old_status} -> {new_status}")
            if old_status != new_status:
                print(f"    ⏰ Timestamps - planned: {tasks[i]['plannedAt']}, started: {tasks[i]['startedAt']}, completed: {tasks[i]['completedAt']}")
    
    return updated_json

def compare_task_names(stored_tasks, new_tasks):
    """
    Compare task names between stored and new tasks to detect considerable changes.
    
    Args:
        stored_tasks: List of tasks from stored JSON
        new_tasks: List of tasks from new OCR result
    
    Returns:
        dict: {
            'has_considerable_changes': bool,
            'changed_tasks': list of dict with 'index', 'old_name', 'new_name', 'change_type'
        }
    """
    import difflib
    
    changed_tasks = []
    
    # Ensure we have the same number of tasks to compare
    min_length = min(len(stored_tasks), len(new_tasks))
    
    for i in range(min_length):
        old_name = stored_tasks[i].get('name', '').strip()
        new_name = new_tasks[i].get('name', '').strip()
        
        if old_name == new_name:
            continue  # No change
        
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, old_name.lower(), new_name.lower()).ratio()
        
        # Detect different types of changes
        change_type = None
        is_considerable = False
        
        if similarity < 0.3:  # Very different
            change_type = "major_change"
            is_considerable = True
        elif similarity < 0.7:  # Moderately different
            change_type = "moderate_change"
            is_considerable = True
        else:
            # Check for added/removed words (minor changes that might still be significant)
            old_words = set(old_name.lower().split())
            new_words = set(new_name.lower().split())
            
            added_words = new_words - old_words
            removed_words = old_words - new_words
            
            # More strict criteria for "considerable" changes
            # Only consider it considerable if:
            # - 2+ words added/removed AND at least one is 4+ characters
            # - OR any very long words (7+ chars) added/removed
            # - OR substantial character difference (more than 3 chars different)
            char_diff = abs(len(old_name) - len(new_name))
            
            considerable_word_changes = (
                (len(added_words) >= 2 and any(len(word) >= 4 for word in added_words)) or
                (len(removed_words) >= 2 and any(len(word) >= 4 for word in removed_words)) or
                any(len(word) >= 7 for word in added_words) or
                any(len(word) >= 7 for word in removed_words) or
                char_diff > 3
            )
            
            if considerable_word_changes:
                change_type = "word_changes"
                is_considerable = True
            else:
                change_type = "minor_change"
                is_considerable = False
        
        if is_considerable:
            changed_tasks.append({
                'index': i,
                'old_name': old_name,
                'new_name': new_name,
                'change_type': change_type,
                'similarity': similarity
            })
    
    # Check for length differences (added/removed tasks)
    if len(stored_tasks) != len(new_tasks):
        if len(new_tasks) > len(stored_tasks):
            # New tasks added
            for i in range(len(stored_tasks), len(new_tasks)):
                changed_tasks.append({
                    'index': i,
                    'old_name': None,
                    'new_name': new_tasks[i].get('name', '').strip(),
                    'change_type': 'added_task',
                    'similarity': 0.0
                })
        else:
            # Tasks removed (we'll handle this by noting missing tasks)
            for i in range(len(new_tasks), len(stored_tasks)):
                changed_tasks.append({
                    'index': i,
                    'old_name': stored_tasks[i].get('name', '').strip(),
                    'new_name': None,
                    'change_type': 'removed_task',
                    'similarity': 0.0
                })
    
    has_considerable_changes = len(changed_tasks) > 0
    
    return {
        'has_considerable_changes': has_considerable_changes,
        'changed_tasks': changed_tasks
    }

def update_specific_tasks_in_firestore(daily_id, changed_tasks, updated_json):
    """
    Update only specific tasks in Firestore based on the changes detected.
    
    Args:
        daily_id: The daily document ID
        changed_tasks: List of changed tasks from compare_task_names
        updated_json: The complete updated JSON with new data
    """
    try:
        from process_JSON import initialize_firebase
        from firebase_admin import firestore
        
        # Initialize Firebase
        initialize_firebase()
        db = firestore.client()
        
        print(f"🔄 [Turbo] Updating {len(changed_tasks)} specific tasks in Firestore...")
        
        # Get the tasks from updated JSON
        tasks_data = updated_json.get('tasks', [])
        
        for change in changed_tasks:
            task_index = change['index']
            change_type = change['change_type']
            old_name = change['old_name']
            new_name = change['new_name']
            
            if change_type == 'removed_task':
                # Handle removed tasks - we might want to mark them as deleted or just skip
                print(f"⚠️  [Turbo] Task {task_index} was removed: '{old_name}' - skipping update")
                continue
            
            if task_index >= len(tasks_data):
                print(f"⚠️  [Turbo] Task index {task_index} out of range - skipping")
                continue
            
            task_data = tasks_data[task_index]
            
            # Create task document reference
            # Using order field as document ID (similar to process_JSON logic)
            task_order = task_data.get('order', task_index + 1)
            task_doc_id = str(task_order).zfill(3)  # e.g., "001", "002", etc.
            
            task_ref = db.collection('daily').document(daily_id).collection('tasks').document(task_doc_id)
            
            # Prepare update data
            update_data = {
                'name': task_data.get('name', ''),
                'status': task_data.get('status', 'NOT_STARTED'),
                'order': task_order,
                'updatedAt': firestore.SERVER_TIMESTAMP
            }
            
            # Add timestamps if they exist
            if task_data.get('plannedAt'):
                update_data['plannedAt'] = task_data['plannedAt']
            if task_data.get('startedAt'):
                update_data['startedAt'] = task_data['startedAt']
            if task_data.get('completedAt'):
                update_data['completedAt'] = task_data['completedAt']
            if task_data.get('projectRef'):
                update_data['projectRef'] = task_data['projectRef']
            
            # Update the document with error handling
            try:
                if change_type == 'added_task':
                    # Create new document
                    task_ref.set(update_data)
                    print(f"✅ [Turbo] Created new task {task_doc_id}: '{new_name}'")
                else:
                    # Try to update existing document, but handle missing documents gracefully
                    try:
                        task_ref.update(update_data)
                        print(f"✅ [Turbo] Updated task {task_doc_id}: '{old_name}' → '{new_name}' ({change_type})")
                    except Exception as update_error:
                        if "No document to update" in str(update_error) or "404" in str(update_error):
                            # Document doesn't exist, create it instead
                            print(f"⚠️  [Turbo] Document {task_doc_id} doesn't exist, creating it...")
                            task_ref.set(update_data)
                            print(f"✅ [Turbo] Created task {task_doc_id}: '{new_name}' (was missing)")
                        else:
                            # Re-raise other errors
                            raise update_error
            except Exception as e:
                print(f"❌ [Turbo] Error updating task {task_doc_id}: {str(e)}")
                # Don't stop the whole process for one failed task
                continue
        
        print(f"🎉 [Turbo] Successfully updated {len(changed_tasks)} tasks in Firestore")
        
    except Exception as e:
        print(f"❌ [Turbo] Error updating specific tasks in Firestore: {str(e)}")
        import traceback
        traceback.print_exc()

def turbo_process_json_with_ocr_async(new_order, frame_copy):
    """
    Enhanced turbo processing that includes OCR with weaker model and selective updates.
    
    Args:
        new_order: List of detected status order
        frame_copy: Frame copy for OCR processing
    """
    try:
        print("🚀 [Turbo] Starting enhanced turbo processing with OCR comparison...")
        
        # Step 1: Get existing tasks from Firestore for proper timestamp handling
        try:
            from process_JSON import initialize_firebase
            from firebase_admin import firestore
            initialize_firebase()
            db = firestore.client()
            existing_tasks_map = get_all_tasks_map_sync(turbo_daily_id, db)
            print(f"✅ [Turbo] Retrieved {len(existing_tasks_map)} existing tasks from Firestore")
        except Exception as e:
            print(f"⚠️  [Turbo] Could not retrieve existing tasks: {str(e)}")
            existing_tasks_map = None
        
        # Step 2: Update the stored JSON with new order and proper timestamps
        updated_json = update_json_with_new_order(turbo_stored_json, new_order, existing_tasks_map)
        
        if not updated_json:
            print("❌ [Turbo] Failed to update JSON")
            return
        
        # Step 3: Run OCR with weaker model to get fresh task names
        print("🔍 [Turbo] Running OCR with weaker model to check for name changes...")
        ocr_result = run_ocr(frame_copy)
        
        if not ocr_result:
            print("⚠️  [Turbo] OCR failed, proceeding with status-only update...")
            # Fallback to original turbo behavior
            result = process_json(updated_json, turbo_daily_meta, turbo_daily_id)
            if result.success:
                print("🎉 [Turbo] SUCCESS (status-only update)!")
                print(f"[Turbo] {result.message}")
            else:
                print("💥 [Turbo] FAILED!")
                print(f"[Turbo] {result.message}")
            return
        
        # Step 4: Parse OCR result to get fresh task names
        try:
            # Parse JSON from OCR result (similar to process_json_from_ocr_async)
            fresh_json_data = None
            
            try:
                fresh_json_data = json.loads(ocr_result)
                print("✅ [Turbo] Successfully parsed OCR result as direct JSON")
            except json.JSONDecodeError:
                # Try to extract JSON from markdown or find single-line JSON
                lines = ocr_result.split('\n')
                json_content = []
                in_code_block = False
                
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith('```'):
                        if line_stripped == '```json' or line_stripped == '```':
                            in_code_block = True
                            continue
                        elif in_code_block:
                            break
                    if in_code_block:
                        json_content.append(line)
                
                if json_content:
                    json_string = '\n'.join(json_content)
                    try:
                        fresh_json_data = json.loads(json_string)
                        print("✅ [Turbo] Successfully extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        pass
                
                # Try single-line JSON as fallback
                if not fresh_json_data:
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                fresh_json_data = json.loads(line)
                                print("✅ [Turbo] Successfully extracted JSON from single line")
                                break
                            except json.JSONDecodeError:
                                continue
            
            if not fresh_json_data:
                print("⚠️  [Turbo] Could not parse fresh JSON from OCR, proceeding with status-only update...")
                # Fallback to original behavior
                result = process_json(updated_json, turbo_daily_meta, turbo_daily_id)
                if result.success:
                    print("🎉 [Turbo] SUCCESS (status-only update)!")
                    print(f"[Turbo] {result.message}")
                else:
                    print("💥 [Turbo] FAILED!")
                    print(f"[Turbo] {result.message}")
                return
            
        except Exception as e:
            print(f"⚠️  [Turbo] Error parsing fresh OCR result: {str(e)}")
            # Fallback to original behavior
            result = process_json(updated_json, turbo_daily_meta, turbo_daily_id)
            if result.success:
                print("🎉 [Turbo] SUCCESS (status-only update)!")
                print(f"[Turbo] {result.message}")
            else:
                print("💥 [Turbo] FAILED!")
                print(f"[Turbo] {result.message}")
            return
        
        # Step 5: Compare task names between stored and fresh OCR
        stored_tasks = turbo_stored_json.get('tasks', [])
        fresh_tasks = fresh_json_data.get('tasks', [])
        
        comparison_result = compare_task_names(stored_tasks, fresh_tasks)
        
        if comparison_result['has_considerable_changes']:
            print(f"🔍 [Turbo] Considerable name changes detected ({len(comparison_result['changed_tasks'])} tasks):")
            for change in comparison_result['changed_tasks']:
                print(f"   Task {change['index']}: '{change['old_name']}' → '{change['new_name']}' ({change['change_type']})")
            
            # Update the JSON with fresh names
            print("🔄 [Turbo] Updating JSON with fresh task names...")
            # Merge fresh names with updated statuses
            for i, fresh_task in enumerate(fresh_tasks):
                if i < len(updated_json.get('tasks', [])):
                    # Keep the status and timestamps from updated_json, but use fresh name
                    updated_json['tasks'][i]['name'] = fresh_task.get('name', '')
                    # Also update projectRef if it changed
                    if fresh_task.get('projectRef'):
                        updated_json['tasks'][i]['projectRef'] = fresh_task['projectRef']
            
            # Handle added tasks
            if len(fresh_tasks) > len(updated_json.get('tasks', [])):
                for i in range(len(updated_json.get('tasks', [])), len(fresh_tasks)):
                    new_task = fresh_tasks[i].copy()
                    # Set default status and order for new tasks
                    new_task['status'] = 'NOT_STARTED'
                    new_task['order'] = i + 1
                    new_task['plannedAt'] = None
                    new_task['startedAt'] = None
                    new_task['completedAt'] = None
                    updated_json['tasks'].append(new_task)
            
            # Use selective Firestore update
            print("🎯 [Turbo] Using selective Firestore update for changed tasks only...")
            try:
                update_specific_tasks_in_firestore(turbo_daily_id, comparison_result['changed_tasks'], updated_json)
            except Exception as e:
                print(f"⚠️  [Turbo] Selective update failed: {str(e)}")
                print("🔄 [Turbo] Falling back to full process_json update...")
                result = process_json(updated_json, turbo_daily_meta, turbo_daily_id)
                if result.success:
                    print("🎉 [Turbo] SUCCESS (fallback update)!")
                    print(f"[Turbo] {result.message}")
                else:
                    print("💥 [Turbo] FAILED (fallback)!")
                    print(f"[Turbo] {result.message}")
            
        else:
            print("✅ [Turbo] No considerable name changes detected, proceeding with standard update...")
            # Use standard process_json for status-only updates
            result = process_json(updated_json, turbo_daily_meta, turbo_daily_id)
            if result.success:
                print("🎉 [Turbo] SUCCESS!")
                print(f"[Turbo] {result.message}")
            else:
                print("💥 [Turbo] FAILED!")
                print(f"[Turbo] {result.message}")
                
    except Exception as e:
        print(f"❌ [Turbo] Unexpected error in enhanced turbo processing: {str(e)}")
        import traceback
        traceback.print_exc()

def handle_turbo_mode(detections, labels, frame=None):
    """Handle 'w' key press for turbo mode"""
    global turbo_stored_json, turbo_stored_order
    
    if turbo_stored_json is None or turbo_stored_order is None:
        print("⚠️  [Turbo] No stored data found. Please press 'f' first to initialize turbo mode.")
        return
    
    # Get current YOLO order
    current_order = get_yolo_order_top_to_bottom(detections, labels)
    
    print(f"🔍 [Turbo] Stored order: {turbo_stored_order}")
    print(f"🔍 [Turbo] Current order: {current_order}")
    
    # Compare orders
    if current_order == turbo_stored_order:
        print("⚡ [Turbo] Orders match! Using fast path...")
        # Start turbo processing in background thread
        turbo_thread = threading.Thread(target=turbo_process_json_with_ocr_async, args=(current_order, frame), daemon=True)
        turbo_thread.start()
    elif len(current_order) != len(turbo_stored_order):
        print("🔄 [Turbo] Different number of symbols detected. Falling back to full OCR...")
        print("💡 [Turbo] This will update the stored data after processing.")
        if frame is not None:
            frame_copy = frame.copy()
            print("🔍 [Turbo] Starting full OCR processing with turbo storage...")
            # Start full OCR processing in background thread - this will update turbo storage
            ocr_thread = threading.Thread(target=process_json_from_ocr_async_with_turbo_storage, args=(frame_copy, detections, labels), daemon=True)
            ocr_thread.start()
        else:
            print("🚨 [Turbo] No frame available for OCR fallback - please use 'f' key manually")
    else:
        print("⚡ [Turbo] Same count but different order! Using turbo mode with new order...")
        # Update stored order and process
        turbo_stored_order = current_order
        turbo_thread = threading.Thread(target=turbo_process_json_with_ocr_async, args=(current_order, frame), daemon=True)
        turbo_thread.start()

def detect_full_page_view(detections, labels):
    """
    Detect if the page is fully in view and stable.
    Returns True when the page appears to be fully visible and stable.
    """
    global page_view_detector
    
    if detections is None:
        return False
    
    # Get current symbol positions
    current_positions = []
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        
        if conf > CONFIDENCE_THRESHOLD and classname != "TEXT_AREA":
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            current_positions.append((center_x, center_y, classname))
    
    # Add current positions to history
    page_view_detector['position_history'].append(current_positions)
    
    # Need at least 2 frames to compare
    if len(page_view_detector['position_history']) < 2:
        return False
    
    # Check if positions are stable (symbols not moving much)
    prev_positions = page_view_detector['position_history'][-2]
    
    # If the number of symbols is different, reset stability
    if len(current_positions) != len(prev_positions):
        page_view_detector['stable_frames'] = 0
        return False
    
    # Check if positions are stable (within threshold)
    positions_stable = True
    if len(current_positions) > 0 and len(prev_positions) > 0:
        # Match symbols by class name and proximity
        for curr_pos in current_positions:
            curr_x, curr_y, curr_class = curr_pos
            best_match = None
            best_distance = float('inf')
            
            for prev_pos in prev_positions:
                prev_x, prev_y, prev_class = prev_pos
                if curr_class == prev_class:
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = prev_pos
            
            if best_match is None or best_distance > page_view_detector['position_threshold']:
                positions_stable = False
                break
    
    # Update stability counter
    if positions_stable:
        page_view_detector['stable_frames'] += 1
    else:
        page_view_detector['stable_frames'] = 0
    
    # Check if we have enough stable frames
    is_stable = page_view_detector['stable_frames'] >= page_view_detector['required_stable_frames']
    
    return is_stable

def detect_initial_page_ready(detections, labels, frame_shape):
    """
    Detect if a page is ready for initial scan.
    
    This function implements sophisticated logic to determine when a page
    has come into full view for the first time.
    
    Returns True when:
    1. Symbol count has stabilized (stopped growing)
    2. Minimum symbol threshold is met
    3. Symbols are distributed (not all clustered at edges)
    4. The view has been stable for sufficient frames
    """
    global initial_scan_detector
    
    if initial_scan_detector['has_scanned_once']:
        return False  # Already did initial scan
    
    # Reduce cooldown
    if initial_scan_detector['initial_scan_cooldown'] > 0:
        initial_scan_detector['initial_scan_cooldown'] -= 1
        return False
    
    # Get current symbol count and positions
    current_order = get_yolo_order_top_to_bottom(detections, labels)
    current_count = len(current_order)
    
    # Get symbol positions for distribution check
    symbol_positions = []
    if detections is not None:
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()
            
            if conf > CONFIDENCE_THRESHOLD and classname != "TEXT_AREA":
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                symbol_positions.append((center_x, center_y))
    
    # Add to history
    initial_scan_detector['symbol_count_history'].append(current_count)
    
    # Update max count seen
    if current_count > initial_scan_detector['max_symbol_count_seen']:
        initial_scan_detector['max_symbol_count_seen'] = current_count
        initial_scan_detector['growth_stopped_frames'] = 0  # Reset growth timer
    else:
        initial_scan_detector['growth_stopped_frames'] += 1
    
    # Need minimum history to make decisions
    if len(initial_scan_detector['symbol_count_history']) < 10:
        return False
    
    # Check if we meet minimum symbol threshold
    if current_count < initial_scan_detector['min_symbol_threshold']:
        initial_scan_detector['stable_count_frames'] = 0
        return False
    
    # Check if symbol count has been stable
    recent_counts = list(initial_scan_detector['symbol_count_history'])[-10:]
    if len(set(recent_counts)) <= 2:  # Allow small variation
        initial_scan_detector['stable_count_frames'] += 1
    else:
        initial_scan_detector['stable_count_frames'] = 0
    
    # Check if growth has stopped (no new symbols appearing)
    growth_stopped = initial_scan_detector['growth_stopped_frames'] >= initial_scan_detector['growth_stop_threshold']
    
    # Check symbol distribution (not all clustered at edges)
    well_distributed = True
    if symbol_positions:
        frame_height, frame_width = frame_shape[:2]
        edge_margin = initial_scan_detector['edge_margin']
        
        # Count symbols near edges
        edge_symbols = 0
        for x, y in symbol_positions:
            if (x < edge_margin or x > frame_width - edge_margin or
                y < edge_margin or y > frame_height - edge_margin):
                edge_symbols += 1
        
        # If more than 80% of symbols are at edges, consider it partial view
        if len(symbol_positions) > 0 and edge_symbols / len(symbol_positions) > 0.8:
            well_distributed = False
    
    # Check if we meet all conditions
    stable_enough = initial_scan_detector['stable_count_frames'] >= initial_scan_detector['required_stable_frames']
    
    ready_for_scan = (
        stable_enough and 
        growth_stopped and 
        well_distributed and 
        current_count >= initial_scan_detector['min_symbol_threshold']
    )
    
    if ready_for_scan:
        print(f"📄 [Initial] Page ready for first scan!")
        print(f"   - Symbol count: {current_count}")
        print(f"   - Stable frames: {initial_scan_detector['stable_count_frames']}")
        print(f"   - Growth stopped: {initial_scan_detector['growth_stopped_frames']} frames ago")
        print(f"   - Well distributed: {well_distributed}")
        
        # Mark as waiting to prevent multiple triggers
        initial_scan_detector['waiting_for_initial_scan'] = True
        initial_scan_detector['initial_scan_cooldown'] = 180  # 3 second cooldown
        
        return True
    
    return False

def mark_initial_scan_complete():
    """Mark that the initial scan has been completed"""
    global initial_scan_detector
    initial_scan_detector['has_scanned_once'] = True
    initial_scan_detector['waiting_for_initial_scan'] = False
    print("✅ [Initial] Initial scan completed - switching to normal auto mode")

def check_automatic_mode_trigger(detections, labels, frame_shape=None):
    """
    Check if we should automatically trigger turbo mode or wait for full page view.
    Now also handles initial scan detection.
    Returns: 'initial_scan', 'turbo', 'wait_for_full_view', 'full_ocr', or None
    """
    global previous_symbol_count, current_symbol_count, symbol_count_history
    global auto_turbo_cooldown, auto_ocr_cooldown, page_view_detector, initial_scan_detector
    
    if not auto_mode_enabled:
        return None
    
    # Check for initial scan first (if we haven't done one yet)
    if not initial_scan_detector['has_scanned_once'] and not initial_scan_detector['waiting_for_initial_scan']:
        if detect_initial_page_ready(detections, labels, frame_shape):
            return 'initial_scan'
    
    # If we're waiting for initial scan to complete, don't do anything else
    if initial_scan_detector['waiting_for_initial_scan']:
        return None
    
    # Rest of the existing logic for post-initial-scan automation...
    # Reduce cooldowns
    if auto_turbo_cooldown > 0:
        auto_turbo_cooldown -= 1
    if auto_ocr_cooldown > 0:
        auto_ocr_cooldown -= 1
    
    # Get current symbol count
    current_order = get_yolo_order_top_to_bottom(detections, labels)
    current_symbol_count = len(current_order)
    
    # Add to history for stability check
    symbol_count_history.append(current_symbol_count)
    
    # Need stable symbol count to make decisions
    if len(symbol_count_history) < stability_threshold:
        return None
    
    # Check if symbol count has been stable
    recent_counts = list(symbol_count_history)[-stability_threshold:]
    if len(set(recent_counts)) > 1:  # Not all the same
        return None
    
    stable_count = recent_counts[0]
    
    # If no previous baseline, set it
    if previous_symbol_count == 0:
        previous_symbol_count = stable_count
        return None
    
    # Check if we're currently waiting for full view
    if page_view_detector['waiting_for_full_view']:
        page_view_detector['wait_counter'] += 1
        
        # Check if page is now stable
        if detect_full_page_view(detections, labels):
            print("📄 [Auto] Page is now stable and fully in view!")
            page_view_detector['waiting_for_full_view'] = False
            page_view_detector['wait_counter'] = 0
            previous_symbol_count = stable_count  # Update baseline
            return 'full_ocr'
        
        # Timeout - proceed with OCR anyway
        if page_view_detector['wait_counter'] >= page_view_detector['max_wait_time']:
            print("⏰ [Auto] Timeout waiting for stable view, proceeding with OCR...")
            page_view_detector['waiting_for_full_view'] = False
            page_view_detector['wait_counter'] = 0
            previous_symbol_count = stable_count  # Update baseline
            return 'full_ocr'
        
        return 'wait_for_full_view'
    
    # Decision logic
    if stable_count == previous_symbol_count and stable_count > 0:
        # Same number of symbols - use turbo mode
        if auto_turbo_cooldown == 0 and turbo_stored_json is not None:
            print(f"⚡ [Auto] Same symbol count ({stable_count}) detected - triggering TURBO mode")
            auto_turbo_cooldown = 60  # Cooldown frames
            return 'turbo'
    
    elif stable_count > previous_symbol_count:
        # More symbols - wait for full page view
        if auto_ocr_cooldown == 0:
            print(f"📄 [Auto] More symbols detected ({previous_symbol_count} → {stable_count}) - waiting for full page view...")
            page_view_detector['waiting_for_full_view'] = True
            page_view_detector['wait_counter'] = 0
            page_view_detector['stable_frames'] = 0
            auto_ocr_cooldown = 120  # Longer cooldown for OCR
            return 'wait_for_full_view'
    
    elif stable_count < previous_symbol_count and stable_count > 0:
        # Fewer symbols - might be partial view, wait a bit then do OCR
        if auto_ocr_cooldown == 0:
            print(f"📄 [Auto] Fewer symbols detected ({previous_symbol_count} → {stable_count}) - triggering OCR")
            previous_symbol_count = stable_count
            auto_ocr_cooldown = 90  # Cooldown frames
            return 'full_ocr'
    
    return None

def parse_arguments():
    """Parse command line arguments for orientation mode"""
    parser = argparse.ArgumentParser(description='Webcam YOLO Detection with Orientation Control')
    
    orientation_group = parser.add_mutually_exclusive_group()
    orientation_group.add_argument('--phone', action='store_true', 
                                 help='Use phone orientation (90° clockwise rotation) - default')
    orientation_group.add_argument('--landscape', action='store_true',
                                 help='Use landscape orientation (make image vertical)')
    
    args = parser.parse_args()
    
    # Determine orientation mode
    if args.landscape:
        orientation_mode = "landscape"
    else:
        orientation_mode = "phone"  # Default
    
    return orientation_mode

def main():
    global auto_mode_enabled, previous_symbol_count, current_symbol_count
    global auto_turbo_cooldown, auto_ocr_cooldown, page_view_detector, initial_scan_detector
    
    # Parse command line arguments
    orientation_mode = parse_arguments()
    print(f"📱 Using orientation mode: {orientation_mode}")
    
    print("Loading YOLO model...")
    try:
        # Load the YOLO model
        model = YOLO(MODEL_PATH)
        labels = model.names
        print(f"Model loaded successfully. Available classes: {len(labels)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have ultralytics installed: pip install ultralytics")
        return

    # Initialize webcam capture
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Initialize FPS calculation variables
    frame_rate_buffer = []
    fps_avg_len = 30
    avg_frame_rate = 0

    print("Starting webcam with YOLO detection...")
    print(f"📸 FULLY AUTOMATIC MODE: Just show your page to the camera! (Orientation: {orientation_mode})")
    print("Press 'q' to quit, 's' to pause, 'p' to save screenshot")
    print("Press 'a' for manual OCR, 'f' for manual JSON processing, 'w' for manual Turbo mode")
    print("Press 'z' to toggle AUTO mode (currently ON)")

    while True:
        t_start = time.perf_counter()
        
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Apply image transformations
        transformed_frame = apply_image_transformations(frame, orientation_mode)
        frame_shape = transformed_frame.shape

        # Run YOLO inference on the frame
        results = model(transformed_frame, verbose=False)
        detections = results[0].boxes

        # Initialize object counter
        object_count = 0

        # Process each detection
        if detections is not None:
            for i in range(len(detections)):
                # Get bounding box coordinates
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                # Get class ID and name
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]

                # Get confidence score
                conf = detections[i].conf.item()

                # Draw bounding box if confidence is above threshold and class is not TEXT_AREA
                if conf > CONFIDENCE_THRESHOLD and classname != "TEXT_AREA":
                    color = label_colors.get(classname, (255, 255, 255))  # Default to white if label not found
                    
                    # Draw bounding box
                    cv2.rectangle(transformed_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    # Prepare label text
                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    
                    # Draw label background
                    cv2.rectangle(transformed_frame, (xmin, label_ymin-labelSize[1]-10), 
                                (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    
                    # Draw label text
                    cv2.putText(transformed_frame, label, (xmin, label_ymin-7), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    object_count += 1

        # CHECK AUTOMATIC MODE TRIGGER
        if auto_mode_enabled:
            auto_action = check_automatic_mode_trigger(detections, labels, frame_shape)
            
            if auto_action == 'initial_scan':
                frame_copy = transformed_frame.copy()
                print("🎯 [Auto] Triggering INITIAL SCAN - first time setup...")
                json_thread = threading.Thread(target=lambda: [
                    process_json_from_ocr_async_with_turbo_storage(frame_copy, detections, labels),
                    mark_initial_scan_complete()
                ], daemon=True)
                json_thread.start()
            elif auto_action == 'turbo':
                handle_turbo_mode(detections, labels, transformed_frame)
            elif auto_action == 'full_ocr':
                frame_copy = transformed_frame.copy()
                print("🤖 [Auto] Triggering full OCR processing...")
                json_thread = threading.Thread(target=process_json_from_ocr_async_with_turbo_storage, args=(frame_copy, detections, labels), daemon=True)
                json_thread.start()

        # Calculate and display FPS
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))
        
        # Update FPS buffer
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

        # Draw FPS and object count on frame
        cv2.putText(transformed_frame, f'FPS: {avg_frame_rate:.1f}', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(transformed_frame, f'Objects detected: {object_count}', (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display AUTO mode status
        auto_status = "FULLY AUTO" if auto_mode_enabled else "MANUAL"
        auto_color = (0, 255, 0) if auto_mode_enabled else (0, 0, 255)
        cv2.putText(transformed_frame, auto_status, (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, auto_color, 2)
        
        # Display initial scan status
        if not initial_scan_detector['has_scanned_once']:
            if initial_scan_detector['waiting_for_initial_scan']:
                init_text = "Processing first scan..."
                init_color = (0, 255, 255)  # Yellow
            else:
                current_count = len(get_yolo_order_top_to_bottom(detections, labels))
                stable_frames = initial_scan_detector['stable_count_frames']
                required_frames = initial_scan_detector['required_stable_frames']
                
                init_text = f"Waiting for page: {current_count} symbols, {stable_frames}/{required_frames} stable"
                init_color = (255, 255, 0)  # Cyan
            
            cv2.putText(transformed_frame, init_text, (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, init_color, 1)
        
        # Display waiting status if applicable
        elif page_view_detector['waiting_for_full_view']:
            wait_text = f"Waiting for stable view... ({page_view_detector['wait_counter']}/{page_view_detector['max_wait_time']})"
            cv2.putText(transformed_frame, wait_text, (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display cooldown status
        if auto_turbo_cooldown > 0 or auto_ocr_cooldown > 0:
            cooldown_text = f"Cooldown: T{auto_turbo_cooldown} O{auto_ocr_cooldown}"
            cv2.putText(transformed_frame, cooldown_text, (10, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Display the frame
        cv2.imshow('Webcam YOLO Detection (press q to quit)', transformed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Paused - press any key to continue...")
            cv2.waitKey(0)
        elif key == ord('p'):
            screenshot_name = f'screenshot_{int(time.time())}.png'
            cv2.imwrite(screenshot_name, transformed_frame)
            print(f"Screenshot saved as {screenshot_name}")
        elif key == ord('a'):
            # Create a copy of the frame for the background thread
            frame_copy = transformed_frame.copy()
            print("🔍 A key pressed - Starting OCR in background...")
            # Start OCR in a separate thread so it doesn't block the main loop
            ocr_thread = threading.Thread(target=perform_ocr_async, args=(frame_copy,), daemon=True)
            ocr_thread.start()
        elif key == ord('f'):
            # Create a copy of the frame for the background thread
            frame_copy = transformed_frame.copy()
            print("🔥 F key pressed - Starting JSON processing with turbo storage in background...")
            # Start JSON processing in a separate thread so it doesn't block the main loop
            json_thread = threading.Thread(target=process_json_from_ocr_async_with_turbo_storage, args=(frame_copy, detections, labels), daemon=True)
            json_thread.start()
        elif key == ord('w'):
            handle_turbo_mode(detections, labels, transformed_frame)
        elif key == ord('r'):
            # Reset initial scan detector (for testing)
            initial_scan_detector['has_scanned_once'] = False
            initial_scan_detector['waiting_for_initial_scan'] = False
            initial_scan_detector['symbol_count_history'].clear()
            initial_scan_detector['max_symbol_count_seen'] = 0
            initial_scan_detector['stable_count_frames'] = 0
            initial_scan_detector['growth_stopped_frames'] = 0
            print("🔄 [Reset] Initial scan detector reset - ready for new first scan")
        elif key == ord('z'):
            # Toggle AUTO mode
            auto_mode_enabled = not auto_mode_enabled
            status = "enabled" if auto_mode_enabled else "disabled"
            print(f"🤖 AUTO mode {status}")
            if not auto_mode_enabled:
                # Reset state when disabling
                page_view_detector['waiting_for_full_view'] = False
                page_view_detector['wait_counter'] = 0
                auto_turbo_cooldown = 0
                auto_ocr_cooldown = 0

    # Clean up
    print(f"Average FPS: {avg_frame_rate:.2f}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 