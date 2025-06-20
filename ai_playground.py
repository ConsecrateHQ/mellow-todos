import google.generativeai as genai
from PIL import Image
from projects_playground import get_all_projects
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

# Configure the generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# Set generation configuration
generation_config = {
    "temperature": 0.1,  # Lower temperature for more consistent output
    "top_p": 1,
    "top_k": 1,
   #  "max_output_tokens": 2048,
}

for model in genai.list_models():
    print(model.name)

# Initialize the model
# gemini-2.5-pro-preview-06-05
# gemini-2.5-flash-preview-05-20
# models/gemini-2.5-flash-preview-04-17-thinking
# models/gemini-2.5-flash-lite-preview-06-17
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-lite-preview-06-17",
    generation_config=generation_config,
)

def run_ocr(image):
    """
    Perform OCR on an image using Gemini AI with TODO list analysis
    
    Args:
        image: PIL Image object or OpenCV frame (BGR format)
    
    Returns:
        str: Extracted and analyzed TODO list as JSON, or None if error occurred
    """
    try:
        # If it's an OpenCV frame (numpy array), convert to PIL Image
        if hasattr(image, 'shape') and len(image.shape) == 3:
            # Assume it's OpenCV format (BGR)
            import cv2
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
        else:
            # Assume it's already a PIL Image
            pil_image = image
        
        # Send to Gemini for OCR using the advanced TODO analysis prompt with projects data
        updated_prompt = get_updated_prompt()
        response = model.generate_content([updated_prompt, pil_image])
        return response.text
        
    except Exception as e:
        print(f"Error performing OCR with Gemini: {e}")
        return None

def get_projects_for_prompt():
    """
    Query the database for projects and format them for the prompt
    
    Returns:
        str: Formatted string with project information to append to the prompt
    """
    try:
        projects = get_all_projects()
        
        if not projects:
            return "No projects found in the database."
        
        project_list = []
        for project in projects:
            project_name = project.get('name', 'Unknown')
            project_description = project.get('description', 'No description')
            project_list.append(f"- {project_name}: {project_description}")
        
        return "\n".join(project_list)
        
    except Exception as e:
        print(f"Error retrieving projects: {e}")
        return "Error retrieving projects from database."

def get_updated_prompt():
    """
    Get the prompt with projects data appended
    
    Returns:
        str: Complete prompt with project information
    """
    projects_info = get_projects_for_prompt()
    return new_prompt_3 + "\n" + projects_info

new_prompt_3 = """
You are receiving an image of a handwritten TODO list where YOLO object detection has labeled various TODO items with symbols and status percentages.

Please analyze this image and:
1. Perform OCR on the handwritten text to extract each TODO item.
2. Some tasks can go onto multiple lines. Make sure to correctly extract the **full text of each task**, combining lines as needed so that multi-line tasks are captured as a single task.
3. Map the detected symbols/statuses to the appropriate status values:
   - Items with "IN_PROGRESS" labels, orange color (#FFA500), should map to "IN_PROGRESS"
   - Items with "NOT_STARTED" labels, red color (#0000FF), should map to "NOT_STARTED"
   - Items with "MEETING" labels, blue color (#00A5FF), should map to "MEETING"
   - Items with "COMPLETED" labels, appear completed, green color (#00FF00), should map to "COMPLETED"
4. If a TODO item is indented further to the right compared to the previous item, treat it as a subtask of the nearest less-indented task above. Subtasks can themselves have their own subtasks. In the JSON output, include a "subtasks" array for any task that has subtasks. If a task has no subtasks, do **not** include the "subtasks" key for that object.
5. **Do not be too careful when deciding if something is a subtask.** If the "Status" symbol or checkbox is not clearly indented compared to the previous task, it is better to assume it is a parent-level task rather than a subtask.
6. For each task and subtask, **best guess which project it belongs to** based on its content and context. Choose from the list of provided project references at the end of this prompt. Populate the `projectRef` field accordingly. All subtasks should share the same `projectRef` as their parent task.
7. If no suitable project can be identified for a task, leave the `projectRef` field as `null`.
8. Return **ONLY** a JSON object in this exact format (no additional text):

{
  "tasks": [
    {
      "name": "extracted task text here",
      "status": "NOT_STARTED|IN_PROGRESS|MEETING|COMPLETED",
      "plannedAt": null,
      "startedAt": null,
      "completedAt": null,
      "order": 1,
      "projectRef": "project_id_here_or_null",
      "subtasks": [
        {
          "name": "subtask text here",
          "status": "NOT_STARTED|IN_PROGRESS|MEETING|COMPLETED",
          "plannedAt": null,
          "startedAt": null,
          "completedAt": null,
          "order": 1,
          "projectRef": "same_project_id_as_parent_or_null"
        }
      ]
    }
  ]
}

**CRITICAL:** For ALL date/time fields (plannedAt, startedAt, completedAt), use ONLY the JSON value `null` (not the string "null", not "N/A", not any other string). The system expects null values for unset timestamps.

Extract each visible TODO item from the handwritten list, preserving the original text as much as possible (including multi-line tasks), and assign the appropriate status based on the YOLO detection labels or color coding shown in the image. Represent task hierarchy accurately using the "subtasks" structure, but **only create subtasks if the indentation is clear and obvious.**

## Available Projects:
"""

if __name__ == "__main__":
    # Load the image
    image_path = "screenshot_1749877867.png"
    image = Image.open(image_path)

    for model_info in genai.list_models():
        print(model_info.name)
    # Generate content with the image
    response = model.generate_content([get_updated_prompt(), image])

    # Debug the response
    # print("Response object:", response)
    # print("Candidates:", response.candidates)
    # if response.candidates:
    #     for i, candidate in enumerate(response.candidates):
    #         print(f"Candidate {i}:")
    #         print(f"  Finish reason: {candidate.finish_reason}")
    #         print(f"  Safety ratings: {candidate.safety_ratings}")
    #         print(f"  Content: {candidate.content}")
    #         if candidate.content and candidate.content.parts:
    #             print(f"  Parts: {candidate.content.parts}")
    #         else:
    #             print("  No content parts available")

    # Only try to print text if we have valid content
    try:
        print("\nResponse text:")
        print(response.text)
    except ValueError as e:
        print(f"\nError accessing response.text: {e}")
        print("This usually means the response was blocked or incomplete.")