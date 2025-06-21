from flask import Flask, request, jsonify, send_file, make_response, render_template, redirect, session, url_for
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
from database import get_db_connection
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date
from calendar import monthrange
from PIL import Image
from collections import defaultdict


#Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

CORS(app, supports_credentials=True)  # Enable CORS for all routes


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    elif request.method == 'POST':
        data = request.get_json()
        email = data['email']
        password = data['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user[3], password):
            #session here
            user_role = user[4]  #as 5th col in table
            user_name = user[1]  #As 2nd col in table
            user_id = user[0]     #as 1st col in table

            #Save in seesion
            session['email'] = user[2]
            session['username'] = user_name
            session['role'] = user_role
            session['user_id'] = user_id

            if user_role == 'admin':
                return jsonify({"redirect": url_for('admin_dashboard')})
            else:
                return jsonify({"redirect": url_for('dashboard')})     
        else:
            return jsonify({"error": "Invalid email or password"}), 401

#Registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')  # Show the form

    elif request.method == 'POST':
        data = request.get_json()
        username = data['username']
        email = data['email']
        password = data['password']
        role = data['role'] 

        conn = get_db_connection()
        cursor = conn.cursor()

         # Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            conn.close()
            return jsonify({"error": "User already exists!"}), 400

        hashed_password = generate_password_hash(password)

        cursor.execute("INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
                       (username, email, hashed_password, role))
        conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"message": "User registered successfully!"})


#Student Dashboard
@app.route('/dashboard')
def dashboard():
    #print("Session:", session)
    if session.get('role') != 'student':
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

#Logout for session
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


#Admin Dashboard
@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html', username=session.get('username'))
    
#Store user results in db
@app.route('/store_result', methods=['POST'])
def store_result():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    food_items = data['items']
    total_calories = data['total_calories']
    timestamp = datetime.now()
    image_path = data['image_path']         
    result_image_path = data['result_image_path'] 

    user_id = session['user_id']
    today = date.today()

    # Shorten image path
    image_filename = image_path.split('/')[-1]
    result_image_filename = result_image_path.split('/')[-1]

    # Just store short relative paths
    short_image_path = f"/short/{image_filename}"
    short_result_path = f"/short/{result_image_filename}"

    print("Image Path:", short_image_path)
    print("Length of Image Path:", len(short_image_path))

    conn = get_db_connection()
    cursor = conn.cursor()

    #Check if user has already uploaded a meal today
    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE user_id = %s AND DATE(date) = %s
    """, (user_id, today))

    count = cursor.fetchone()[0]

    if count >= 1:
        conn.close()
        return jsonify({"message": "You can only upload one meal per day."}), 409
    
    #Sum macros (carbs, proteins, fats)
    total_carbs = 0
    total_proteins = 0
    total_fats = 0

    for food_name in food_items:
        cursor.execute(
            "SELECT carbs, proteins, fats FROM food_nutrition WHERE food_name = %s",
            (food_name,)
        )
        nutrition = cursor.fetchone()
        if nutrition:
            total_carbs += nutrition[0] or 0
            total_proteins += nutrition[1] or 0
            total_fats += nutrition[2] or 0

    cursor.execute(
        "INSERT INTO predictions (user_id, food_name, calories, protein, carbs, fats, date, image_path, result_image_path) VALUES (%s, %s, %s, %s, %s, %s,%s,%s,%s)",
        (session['user_id'], ','.join(food_items), total_calories, total_proteins, total_carbs, total_fats, timestamp, short_image_path, short_result_path)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Result stored successfully!"})

#Monthly calorie consumption
@app.route('/get_monthly_details', methods=['GET'])
def get_monthly_details():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    today = datetime.now()

    # Accept optional month and year from query params
    month = request.args.get('month', type=int)
    year = request.args.get('year', type=int)

    # Default to current month/year if not provided
    today = datetime.now()
    if not month:
        month = today.month
    if not year:
        year = today.year

    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, monthrange(year, month)[1])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT SUM(calories)
        FROM predictions
        WHERE user_id = %s
          AND MONTH(date) = %s
          AND YEAR(date) = %s
    """, (user_id, month, year))
    total = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT DATE(date), food_name, calories 
        FROM predictions 
        WHERE user_id = %s AND DATE(date) BETWEEN %s AND %s
        ORDER BY date ASC
    """, (user_id, first_day.date(), last_day.date()))

    rows = cursor.fetchall()
    
    conn.close()

    # Format month name (e.g., "April 2025")
    formatted_month = datetime(year, month, 1).strftime("%B %Y")

    records = [
        {
            "date": row[0].strftime('%Y-%m-%d'),
            "food_name": row[1],
            "calories": row[2]
        }
        for row in rows
    ]

    return jsonify({
        "month": formatted_month,
        "monthly_total_calories": total,
        "records": records
    })


#Admin logs of all students
@app.route('/admin/logs', methods=['GET'])
def admin_logs():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, food_name, calories, date
        FROM predictions
        ORDER BY date DESC
    """)

    logs = cursor.fetchall()
    conn.close()

    log_data = [
        {
            "user_id": row[0],
            "food_name": row[1],
            "calories": row[2],
            "timestamp": row[3].strftime("%Y-%m-%d %H:%M:%S")
        }
        for row in logs
    ]

    return jsonify({"logs": log_data})

# Admin attendance API
@app.route('/admin/attendance', methods=['GET'])
def admin_attendance():
    filter_type = request.args.get('filter', 'today')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all users
    cursor.execute("SELECT id FROM users")  # Assuming your users table is named 'users'
    all_users = [row[0] for row in cursor.fetchall()]

    # Fetch active users based on filter
    if filter_type == 'today':
        today = datetime.now().date()
        cursor.execute("""
            SELECT DISTINCT user_id
            FROM predictions
            WHERE DATE(date) = %s
        """, (today,))
    elif filter_type == 'month':
        today = datetime.now()
        first_day = today.replace(day=1)
        cursor.execute("""
            SELECT DISTINCT user_id
            FROM predictions
            WHERE date >= %s
        """, (first_day,))

    active_users = [row[0] for row in cursor.fetchall()]
    conn.close()

    present = len(active_users)
    absent = len(set(all_users) - set(active_users))

    return jsonify({"present": present, "absent": absent})


# /admin/student-access
@app.route('/admin/student-access')
def student_access_summary():
    month = request.args.get('month', datetime.now().strftime("%m"))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT u.id, u.username, COUNT(p.id) as prediction_count
    FROM users u
    JOIN predictions p ON u.id = p.user_id
    WHERE EXTRACT(MONTH FROM p.date) = %s
    GROUP BY u.id, u.username
    """, (month,))
    students = [{"id": row[0], "name": row[1], "count": row[2]} for row in cursor.fetchall()]

    conn.close()

    return jsonify(students)

# /admin/student/<id>/calories
@app.route('/admin/student/<int:student_id>/calories')
def student_calorie_chart(student_id):
    month = request.args.get('month', datetime.now().strftime("%m"))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DATE(date), SUM(calories)
        FROM predictions
        WHERE user_id = %s AND EXTRACT(MONTH FROM date) = %s
        GROUP BY DATE(date)
        ORDER BY DATE(date)
    """, (student_id, month))

    results = cursor.fetchall()
    conn.close()

    data = {
        "dates": [row[0].strftime("%Y-%m-%d") for row in results],
        "calories": [row[1] for row in results]
    }
    return jsonify(data)

# Load both models
v5_model = YOLO("v5_best.pt")
v8_model = YOLO("best.pt")

# Class labels
class_names = ['Puri', 'Bhaji', 'Salad', 'Pickle', 'Chutney', 'Rice', 'Aloo matar bhaji', 'Aloo Bhaji', 
               'Chole Bhaji', 'Cauliflower Bhaji', 'Curd', 'Dal', 'Roti', 'Palak Bhaji', 
               'Paneer Bhaji', 'Papad', 'Dessert']

# Define folder to save processed images
PROCESSED_IMG_DIR = r"G:\My Drive\processed_images"
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

# Path to your labels root folder (adjust as per your project structure)
LABELS_ROOT_DIR = r"G:\My Drive\Paste your annotated files here1\labels"

def load_ground_truth_boxes(image_name, img_width=None, img_height=None):
     # Extract dataset type (train, val, test) from the path
    parts = image_name.replace("\\", "/").split("/")
    if "train" in parts:
        dataset_type = "train"
    elif "val" in parts:
        dataset_type = "val"
    elif "test" in parts:
        dataset_type = "test"
    else:
        print(f"Could not determine dataset type from path: {image_name}")
        return []

    # Extract image filename without extension
    filename = os.path.splitext(os.path.basename(image_name))[0] + ".txt"
    
    # Build label path: Paste Your Annotated Files Here1/labels/train/filename.txt
    label_path = os.path.join(LABELS_ROOT_DIR, dataset_type, filename)

    gt_boxes = []
    if not os.path.exists(label_path):
        print(f"Ground truth label file not found: {label_path}")
        return gt_boxes
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)
           # x_center, y_center, w, h = map(float, (x_center, y_center, w, h))

            # Convert normalized YOLO format to absolute pixel xyxy
            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)

            gt_boxes.append((class_id, [x1, y1, x2, y2]))

    return gt_boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

#Prediction of input image
@app.route('/predict', methods=['POST']) 
def predict():
    #Session holding for user logged in
    user_id = session.get('user_id') 
    
    if user_id:
        today = date.today()

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE user_id = %s AND DATE(date) = %s
        """, (user_id, today))

        count = cursor.fetchone()[0]
        if count >= 1:
            conn.close()
            return jsonify({"message": "You can only upload one meal per day."}), 409
    else:
        conn = get_db_connection()
        cursor = conn.cursor()

    #Check for image input from user
    if 'image' not in request.files:
        cursor.close()
        conn.close()
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    original_filename = secure_filename(image_file.filename)
    image_path = os.path.join(PROCESSED_IMG_DIR, "input.jpg")
    image_file.save(image_path)  #Save input image

    # Predict using both models
    res_v5 = v5_model.predict(image_path, conf=0.25, verbose=False)[0]
    res_v8 = v8_model.predict(image_path, conf=0.25, verbose=False)[0]

    # Load image for size and drawing
    img = Image.open(image_path)
    W, H = img.size

    #dataset_type = os.path.basename(os.path.dirname(original_filename))  # 'train', 'val', or 'test'
    
    # Load ground truth boxes for this image 
    ground_truth_boxes = load_ground_truth_boxes(original_filename,img_width=W, img_height=H)

    # Inference only – no label file for ground truth, so we assume prediction only
    pred_boxes_all = []
    for res, model_name in [(res_v5, "YOLOv5"), (res_v8, "YOLOv8")]:
        for box in res.boxes:
            cls = int(box.cls.item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # keep float for accurate IoU
            conf = box.conf.item()
            pred_boxes_all.append((cls, xyxy, conf, box, model_name))
            print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")

    #Sort predictions by confidence descending
    pred_boxes_all = sorted(pred_boxes_all, key=lambda x: x[2], reverse=True)

     # Now, compare predictions with GT boxes
    iou_threshold = 0.5
    matched_predictions = []  # store tuples of (cls, box, conf, model_name)

    if ground_truth_boxes:  # ✅ Case 1: Ground truth available — match predictions with GT
        used_gt_indices = set()
        # Collect predictions by model
        predictions_by_model = {
            "YOLOv5": [(int(b.cls.item()), b.xyxy[0].cpu().numpy().tolist(), b.conf.item()) for b in res_v5.boxes],
            "YOLOv8": [(int(b.cls.item()), b.xyxy[0].cpu().numpy().tolist(), b.conf.item()) for b in res_v8.boxes]
        }

    #used_gt_indices = set()

        for i, (gt_cls, gt_box) in enumerate(ground_truth_boxes):
            best_iou = 0
            best_match = None
            best_model = None

            for model_name, predictions in predictions_by_model.items():
                for cls_pred, box_pred, conf in predictions:
                    if cls_pred != gt_cls:
                        continue
                    iou = compute_iou(box_pred, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match = (cls_pred, box_pred, conf)
                        best_model = model_name

            if best_match:
                matched_predictions.append((best_match[0], best_match[1], best_match[2], best_model))
                used_gt_indices.add(i)
                print(f"GT {i} matched by {best_model} with IoU {best_iou:.2f}")
            else:
                print(f"GT {i} not matched by any model.")

    else:  # ✅ Case 2: No ground truth — unseen image uploaded by user
        selected_classes = set()
        final_predictions = []

        for cls, box, conf, box_obj, model_name in pred_boxes_all:
            # Check if this box overlaps significantly with any box already in final_predictions
            overlap = False
            for _, existing_box, _, _ in final_predictions:
                if compute_iou(box, existing_box) > iou_threshold:
                    overlap = True
                    break

            # Keep only if there's no significant overlap (even if same class appears multiple times)
            if not overlap:
                final_predictions.append((cls, box, conf, model_name))

            matched_predictions = final_predictions  # Use this for calorie calculation

    total_calories = 0
    detected_items = []  ##List fro detected items

    # # Load image for processing
    orig_image = cv2.imread(image_path)  # Load fresh image
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

    # Process YOLO results
    for cls, box, conf, model_name in matched_predictions:
        x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
        label = class_names[cls] # Get class name
        confidence = conf  # Confidence score

        detected_items.append(label) #Labels detected added to the items list

        # Get calorie value Fetch calories from database for each detected label
        cursor.execute("""
            SELECT calories_per_unit FROM food_nutrition WHERE food_name = %s
        """, (label,))
        calorie_row = cursor.fetchone()

        #Calculate the calories for each food item detected
        calorie = calorie_row[0] if calorie_row else 0
        total_calories += calorie

        # Draw bounding box with thin lines
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box, thickness=2

        # Put label with calories
        text = f"{label}: {calorie} kcal"
        cv2.putText(orig_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Save processed image
    processed_image_path = os.path.join(PROCESSED_IMG_DIR, "output.jpg")
    cv2.imwrite(processed_image_path, cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR))

    # Generate full URL for the processed image
    full_image_url = request.host_url + "processed_images/output.jpg"

    cursor.close()
    conn.close()

    # Return processed image URL and total calorie count
    return jsonify({
        "total_calories": total_calories,
        "processed_image_url": full_image_url,
        "detected_items": list(set(detected_items))  # To avoid duplicates
    })

#Fetch nutritional macro values (proteins. fats, carbs) for Pie chart 
@app.route('/get_nutrition', methods=['GET'])
def get_nutrition():
    food_item = request.args.get('food_name')
    if not food_item:
        return jsonify({"error": "Missing food_name parameter"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM food_nutrition WHERE food_name = %s"
    cursor.execute(query, (food_item,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "Food item not found"}), 404

#Load processed image to frontend for results
@app.route('/processed_images/<filename>')
def get_processed_image(filename):
    processed_image_path = os.path.join(PROCESSED_IMG_DIR, filename)
    if os.path.isfile(processed_image_path):
        response = make_response(send_file(processed_image_path, mimetype='image/jpeg'))
        response.headers['Access-Control-Allow-Origin'] = '*'  # Allow frontend access
        return response
    return "File not found", 404 # Handle missing files

if __name__ == '__main__':
    app.run()
  
