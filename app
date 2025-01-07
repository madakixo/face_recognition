# main app @jamaludeen madaki jan-2025
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils.face_detection import detect_faces
from utils.face_recognition import recognize_faces
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_unique_filename(filename):
    """Generate a unique filename to avoid overwriting issues."""
    base, ext = os.path.splitext(filename)
    unique_id = str(uuid.uuid4())
    return f"{base}_{unique_id}{ext}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = create_unique_filename(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Image Preprocessing
            img = np.array(Image.open(filepath))
            
            # Face Detection (YOLO)
            faces_info = detect_faces(img)  # Get bounding boxes for detected faces
            if faces_info:
                # Face Recognition
                recognized_faces = recognize_faces(img, faces_info)

                # Annotate the image
                annotated_img = annotate_image(img, recognized_faces)
            
            else:
                annotated_img = img

            # Convert annotated_img back to a PIL Image for saving
            annotated_pil_img = Image.fromarray(annotated_img)
            # Save the annotated image
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + unique_filename)
            annotated_pil_img.save(annotated_filepath)
            # Pass the path to the result on to the HTML template
            return render_template('index.html', image_path=annotated_filepath)
    return render_template('index.html')

def annotate_image(img, recognized_faces):
    # Convert to PIL for drawing
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    for (x, y, w, h), label in recognized_faces:
        draw.rectangle(((x, y), (x+w, y+h)), outline=(255, 0, 0), width=2)
        # Draw text background
        text_width, text_height = draw.textsize(label)
        draw.rectangle(((x, y - text_height), (x + text_width, y)), fill=(255, 0, 0))
        # Draw text
        draw.text((x, y - text_height), label, fill=(255, 255, 255))
    return np.array(img_pil)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
