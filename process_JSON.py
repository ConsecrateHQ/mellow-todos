import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import urllib.parse
import pytz
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FirebaseOperationResult:
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data

def initialize_firebase():
    """Initialize Firebase Admin SDK if not already initialized"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
        print("🔥 Firebase already initialized")
    except ValueError:
        # Get Firebase credentials path from environment variable
        credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
        if not credentials_path:
            raise ValueError("FIREBASE_CREDENTIALS_PATH environment variable not set")
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Firebase credentials file not found: {credentials_path}")
        
        # Initialize Firebase Admin SDK with service account key
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        print("🔥 Firebase Admin SDK initialized successfully!")

def to_timestamp(date_input: Any) -> Optional[datetime]:
    """Convert various date formats to datetime object"""
    if not date_input:
        return None
    
    if isinstance(date_input, datetime):
        return date_input
    
    if isinstance(date_input, str):
        # Handle legacy "N/A" values
        if date_input == "N/A":
            return None
            
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_input.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try standard datetime format
                return datetime.strptime(date_input, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    # Try date only format
                    return datetime.strptime(date_input, '%Y-%m-%d')
                except ValueError:
                    # If all parsing fails, return None instead of raising error
                    print(f"⚠️  Warning: Could not parse date '{date_input}', returning None")
                    return None
    
    return None

def task_key(task: Dict[str, Any], parent_name: Optional[str] = None) -> str:
    """Creates a deterministic key for tasks/subtasks"""
    task_name = task.get('name', '')
    return f"{parent_name}::{task_name}" if parent_name else task_name

def key_to_firestore_id(key: str) -> str:
    """Encodes key for use as Firestore doc ID (handles spaces/special chars)"""
    return urllib.parse.quote(key, safe='')

def parse_time_from_task_name(task_name: str, current_date: datetime) -> Optional[datetime]:
    """
    Parse time from task name and return datetime object for today with that time.
    Returns None if no time is found.
    
    Examples:
    - "6:30 pm - Counseling session" -> datetime object for today at 6:30 PM
    - "9:00 am - Team meeting" -> datetime object for today at 9:00 AM
    - "Meeting with client" -> None (no time found)
    """
    if not task_name:
        return None
    
    # Pattern to match time formats like "6:30 pm", "9:00 am", "14:30", etc.
    time_patterns = [
        r'(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)',  # 12-hour format with am/pm
        r'(\d{1,2}):(\d{2})',  # 24-hour format or 12-hour without am/pm
        r'(\d{1,2})\s*(am|pm|AM|PM)',  # Hour only with am/pm
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, task_name)
        if match:
            try:
                if len(match.groups()) == 3:  # 12-hour format with am/pm
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    period = match.group(3).lower()
                    
                    # Convert to 24-hour format
                    if period == 'pm' and hour != 12:
                        hour += 12
                    elif period == 'am' and hour == 12:
                        hour = 0
                        
                elif len(match.groups()) == 2:  # 24-hour format or hour only
                    if ':' in match.group(0):  # Has minutes
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                    else:  # Hour only with am/pm
                        hour = int(match.group(1))
                        minute = 0
                        period = match.group(2).lower()
                        
                        # Convert to 24-hour format
                        if period == 'pm' and hour != 12:
                            hour += 12
                        elif period == 'am' and hour == 12:
                            hour = 0
                else:  # 24-hour format without am/pm
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                
                # Create datetime object for today with the parsed time
                meeting_datetime = current_date.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
                
                return meeting_datetime
                
            except (ValueError, IndexError):
                continue
    
    return None

async def get_all_tasks_map(daily_id: str, db: firestore.Client) -> Dict[str, Any]:
    """Loads all tasks for the given dailyId as a flat map"""
    tasks_ref = db.collection('Dailies').document(daily_id).collection('tasks')
    docs = tasks_ref.stream()
    
    task_map = {}
    
    for doc in docs:
        data = doc.to_dict()
        key = task_key(data)
        task_map[key] = data
        
        # Flatten subtasks (one level deep)
        if 'subtasks' in data and isinstance(data['subtasks'], list):
            for subtask in data['subtasks']:
                sub_key = task_key(subtask, data.get('name'))
                task_map[sub_key] = subtask
    
    return task_map

def get_all_tasks_map_sync(daily_id: str, db: firestore.Client) -> Dict[str, Any]:
    """Synchronous version of get_all_tasks_map"""
    tasks_ref = db.collection('Dailies').document(daily_id).collection('tasks')
    docs = tasks_ref.stream()
    
    task_map = {}
    
    for doc in docs:
        data = doc.to_dict()
        key = task_key(data)
        task_map[key] = data
        
        # Flatten subtasks (one level deep)
        if 'subtasks' in data and isinstance(data['subtasks'], list):
            for subtask in data['subtasks']:
                sub_key = task_key(subtask, data.get('name'))
                task_map[sub_key] = subtask
    
    return task_map

def process_task_timestamps(
    task: Dict[str, Any],
    prev_task: Optional[Dict[str, Any]] = None,
    current_time: Optional[datetime] = None
) -> Tuple[datetime, Optional[datetime], Optional[datetime]]:
    """
    Process timestamps for a task based on status transitions.
    
    Args:
        task: Current task data
        prev_task: Previous task data from Firestore (if exists)
        current_time: Current timestamp (if None, will use current time in Asia/Bangkok)
    
    Returns:
        Tuple of (planned_at, started_at, completed_at)
    """
    # Get timezone-aware current time in your local timezone (+07)
    local_tz = pytz.timezone('Asia/Bangkok')  # UTC+7
    now = current_time or datetime.now(local_tz)
    
    # Handle plannedAt: preserve existing for existing tasks, set to now for new tasks
    if prev_task:
        # Existing task: preserve the original plannedAt, don't change it
        existing_planned_at = prev_task.get('plannedAt')
        planned_at = to_timestamp(existing_planned_at) or now
    else:
        # New task: set plannedAt to current timestamp
        planned_at = now
    
    # Handle other timestamps
    started_at = prev_task.get('startedAt') if prev_task else task.get('startedAt')
    started_at = to_timestamp(started_at) if started_at else None
    
    completed_at = prev_task.get('completedAt') if prev_task else task.get('completedAt')
    completed_at = to_timestamp(completed_at) if completed_at else None
    
    # Special handling for MEETING tasks
    task_status = task.get('status', '')
    if task_status == 'MEETING':
        # Check if startedAt was already provided by AI (and preserve it)
        ai_provided_started_at = task.get('startedAt')
        if ai_provided_started_at is not None and ai_provided_started_at != "null":
            # AI provided a startedAt value, don't override it
            started_at = to_timestamp(ai_provided_started_at) if ai_provided_started_at != "N/A" else None
        else:
            # No startedAt from AI, try to parse time from task name
            task_name = task.get('name', '')
            parsed_time = parse_time_from_task_name(task_name, now)
            
            if parsed_time:
                started_at = parsed_time
            else:
                started_at = None
    else:
        # Handle status transitions for non-MEETING tasks
        if not prev_task:
            # New task: handle initial status transitions
            if task.get('status') == 'IN_PROGRESS':
                started_at = now
            if task.get('status') == 'COMPLETED':
                started_at = now
                completed_at = now
        else:
            # Existing task: check for status transitions
            if (not started_at and 
                prev_task.get('status') != 'IN_PROGRESS' and 
                task.get('status') == 'IN_PROGRESS'):
                started_at = now
            
            if (not completed_at and 
                prev_task.get('status') != 'COMPLETED' and 
                task.get('status') == 'COMPLETED'):
                completed_at = now
    
    return planned_at, started_at, completed_at

def process_task_recursive(
    task: Dict[str, Any],
    existing_tasks_map: Dict[str, Any],
    db: firestore.Client,
    daily_id: str,
    parent_name: Optional[str] = None,
    is_top_level: bool = True,
    order: int = 0
) -> Dict[str, Any]:
    """Recursively processes a task (and subtasks), merges with Firestore data, fills timestamps, and writes to Firestore"""
    
    key = task_key(task, parent_name)
    prev_task = existing_tasks_map.get(key)
    
    # Use the extracted timestamp processing function
    planned_at, started_at, completed_at = process_task_timestamps(task, prev_task)
    
    # Compose new task object for Firestore
    firestore_task = {
        'name': task.get('name', ''),
        'status': task.get('status', ''),
        'plannedAt': planned_at,
        'startedAt': started_at,
        'completedAt': completed_at,
        'order': task.get('order', order),
        'projectRef': task.get('projectRef') or (prev_task.get('projectRef') if prev_task else None),
        'subtasks': []
    }
    
    # Write/merge the task in Firestore (only for top-level tasks)
    tasks_collection = None
    doc_id = ""
    
    if is_top_level:
        tasks_collection = db.collection('Dailies').document(daily_id).collection('tasks')
        doc_id = key_to_firestore_id(key)
        tasks_collection.document(doc_id).set(firestore_task, merge=True)
    
    # Process subtasks recursively
    if 'subtasks' in task and isinstance(task['subtasks'], list) and len(task['subtasks']) > 0:
        for i, subtask in enumerate(task['subtasks']):
            processed_subtask = process_task_recursive(
                subtask,
                existing_tasks_map,
                db,
                daily_id,
                task.get('name'),
                False,  # subtasks are not top-level
                i  # order based on subtask array index
            )
            firestore_task['subtasks'].append(processed_subtask)
        
        # Update the task document with latest subtasks (for nesting, only for top-level tasks)
        if is_top_level and tasks_collection and doc_id:
            tasks_collection.document(doc_id).update({
                'subtasks': firestore_task['subtasks']
            })
    
    return firestore_task

def process_json(
    input_json: Dict[str, Any],
    daily_meta: Dict[str, Any],
    daily_id: str,
    db_instance: Optional[firestore.Client] = None
) -> FirebaseOperationResult:
    """
    Main entry point for processing JSON data.
    
    Args:
        input_json: The input JSON (with only "tasks" key) from AI pipeline
        daily_meta: Metadata for the "daily" key (date, createdAt, updatedAt, cardScannedAt)
        daily_id: Date string (e.g., "2025-05-11") for use as Firestore doc ID
        db_instance: Firestore db instance
    
    Returns:
        FirebaseOperationResult: Result of the operation
    """
    try:
        # Initialize Firebase if not already done
        if db_instance is None:
            initialize_firebase()
            db_instance = firestore.client()
        
        # Compose top-level JSON
        final_json = {
            'daily': {**daily_meta},
            'tasks': []
        }
        
        # Prepare daily metadata with proper timestamps
        daily_doc_data = {
            **daily_meta,
            'date': to_timestamp(daily_meta.get('date')),
            'cardScannedAt': to_timestamp(daily_meta.get('cardScannedAt')) if daily_meta.get('cardScannedAt') else None,
            'createdAt': to_timestamp(daily_meta.get('createdAt')),
            'updatedAt': to_timestamp(daily_meta.get('updatedAt'))
        }
        
        # Upsert the "daily" doc
        db_instance.collection('Dailies').document(daily_id).set(daily_doc_data, merge=True)
        
        # Get all existing tasks for this day (as a flat map)
        existing_tasks_map = get_all_tasks_map_sync(daily_id, db_instance)
        
        # Process tasks recursively
        if 'tasks' in input_json and isinstance(input_json['tasks'], list):
            for i, task in enumerate(input_json['tasks']):
                processed = process_task_recursive(
                    task,
                    existing_tasks_map,
                    db_instance,
                    daily_id,
                    None,
                    True,  # top-level tasks
                    i  # order based on task array index
                )
                final_json['tasks'].append(processed)
        
        # Save the complete JSON to pastJSONs subcollection
        local_tz = pytz.timezone('Asia/Bangkok')  # UTC+7
        now = datetime.now(local_tz)
        past_json_id = now.isoformat()
        past_jsons_collection = db_instance.collection('Dailies').document(daily_id).collection('pastJSONs')
        
        past_jsons_collection.document(past_json_id).set({
            'savedAt': now,
            'dailyJSON': final_json
        })
        
        task_count = len(input_json.get('tasks', []))
        return FirebaseOperationResult(
            success=True,
            message=f"✅ Successfully processed JSON data for {daily_id}. Processed {task_count} tasks and saved JSON snapshot.",
            data=final_json
        )
        
    except Exception as e:
        error_message = f"❌ Error processing JSON: {str(e)}"
        print(error_message)
        return FirebaseOperationResult(
            success=False,
            message=error_message
        )

def main():
    """Test the process_json function with sample data"""
    try:
        # Initialize Firebase
        initialize_firebase()
        db = firestore.client()
        
        # Sample input data
        sample_input_json = {
            "tasks": [
                {
                    "name": "Review project requirements",
                    "status": "COMPLETED",
                    "subtasks": [
                        {
                            "name": "Read specification document",
                            "status": "COMPLETED"
                        },
                        {
                            "name": "Create task breakdown",
                            "status": "IN_PROGRESS"
                        }
                    ]
                },
                {
                    "name": "Implement process_JSON in Python",
                    "status": "IN_PROGRESS",
                    "projectRef": "python-playground"
                },
                {
                    "name": "6:30 pm - Counseling session",
                    "status": "MEETING"
                },
                {
                    "name": "Team meeting with no time",
                    "status": "MEETING"
                }
            ]
        }
        
        sample_daily_meta = {
            "date": "2025-01-15",
            "createdAt": datetime.now(pytz.timezone('Asia/Bangkok')).isoformat(),
            "updatedAt": datetime.now(pytz.timezone('Asia/Bangkok')).isoformat(),
            "cardScannedAt": None
        }
        
        daily_id = "2025-01-15"
        
        # Process the JSON
        result = process_json(sample_input_json, sample_daily_meta, daily_id, db)
        
        if result.success:
            print(result.message)
            print(f"📊 Final JSON structure created with {len(result.data['tasks'])} tasks")
        else:
            print(result.message)
            
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")

# Only run main if this file is executed directly (for testing purposes)
if __name__ == "__main__":
    main()
