import os
import json
import pandas as pd
import pyshark
import csv
import asyncio
import re
from collections import defaultdict
from urllib.parse import urlparse
from flask import Flask, render_template, request, send_from_directory,jsonify
from tld import get_tld
import joblib
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # For progress bar (optional)
from same import AdvancedNNModel
from same2 import LSTMModel
from nettt import predict_from_netcsv
from model_prediction import predict_new_data
app = Flask(__name__)

class_names=["safe","safe","not safe","not safe"]

############################################################################
#URL Prediction
def predict_from_csvurl(csv_file, pth_file, class_names, batch_size=32):
    """
    Load a CSV file and a saved PTH model file, then predict the class names.

    Args:
        csv_file (str): Path to the CSV file with input features.
        pth_file (str): Path to the saved PTH file (model weights).
        class_names (list): List of class names (e.g., ['class_1', 'class_2', ...]).
        batch_size (int): Number of samples to process in each batch (default 32).

    Returns:
        predictions (list): List of predicted class names.
    """
    # Step 1: Load the CSV file
    print("Loading input data from CSV...")
    try:
        data = pd.read_csv(csv_file)
        print(data.head())
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

    # Check if the CSV file is empty
    if data.empty:
        raise ValueError(f"CSV file '{csv_file}' is empty.")


    X = torch.tensor(data.values, dtype=torch.float32)  # Convert to tensor

    # Step 2: Prepare the Model
    print("Loading the model...")
    try:
        input_dim = X.shape[1]  # Number of features from CSV
        output_dim = len(class_names)  # Number of classes based on provided class names

        model = AdvancedNNModel(input_dim, output_dim)
        model.load_state_dict(torch.load(pth_file))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model or weights: {e}")

    # Step 3: Create DataLoader for batch processing
    dataset = TensorDataset(X)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Step 4: Predict in batches
    print("Running predictions...")
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs[0]  # Extract input tensor from the tuple (inputs, labels)
            outputs = model(inputs)
            _, batch_predictions = torch.max(outputs, 1)  # Get predicted class index for each sample
            predictions.extend([class_names[pred] for pred in batch_predictions.int().tolist()])  # Map to class names

    # Return predictions as a list of class names
    return predictions

############################################################################
csv_file_path = "cloud_test.csv"
numerical_cols = ['eventVersion']

categorical_cols = [
    'userAgent', 'eventName', 'awsRegion', 'userIdentitytype', 'userIdentityaccountId',
    'userIdentityprincipalId', 'userIdentityarn', 'userIdentityaccessKeyId', 'userIdentityuserName', 'errorCode'
]



# Load the JIT model
loaded_model = torch.jit.load("model_traced.pt")

# Load the scaler
loaded_scaler = joblib.load("scaler.save")
############################################################################
net_labels = ['BENIGN', 'DoS Hulk', 'FTP-Patator', 'PortScan', 'DDoS',
                'DoS Slowhttptest', 'DoS slowloris', 'Web Attack – XSS', 'Bot',
                'DoS GoldenEye', 'SSH-Patator', 'Web Attack – Brute Force',
                'Infiltration', 'Web Attack – SQL Injection', 'Heartbleed']

# Model parameters (update these with your model specifics)
input_size = 9   # The correct input size used during training
hidden_size = 64  # The correct hidden size used during training
num_classes = 15  # The correct number of classes for the output layer

# Create an instance of the model
model_net = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# Load the saved model weights
model_net.load_state_dict(torch.load('net_model.pth'))
model_net.eval()  # Set to evaluation mode
############################################################################
#URL Uploading
# Function to process the URL and extract features
def process_single_url(url):
    # 1. URL Length
    url_len = len(url)

    # 2. Process domain (TLD extraction)
    def process_tld(url):
        try:
            res = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
            return res.parsed_url.netloc
        except:
            return None

    domain = process_tld(url)

    # 3. Count the number of specific characters in URL
    features = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', '"', ',', '//']
    feature_counts = {feature: url.count(feature) for feature in features}

    # 4. Check for abnormal URL pattern (repeating hostname)
    def abnormal_url(url):
        hostname = urlparse(url).hostname
        return 1 if re.search(hostname, url) else 0

    abnormal_url_flag = abnormal_url(url)

    # 5. Check if the URL is using HTTPS
    def httpSecure(url):
        return 1 if urlparse(url).scheme == 'https' else 0

    https_flag = httpSecure(url)

    # 6. Count digits in the URL
    def digit_count(url):
        return sum(1 for char in url if char.isnumeric())

    digit_count_value = digit_count(url)

    # 7. Count letters in the URL
    def letter_count(url):
        return sum(1 for char in url if char.isalpha())

    letter_count_value = letter_count(url)

    # 8. Check if URL is from a shortening service
    def shortening_service(url):
        match = re.search(r'bit\.ly|goo\.gl|t\.co|tinyurl|adf\.ly|url4\.eu|short\.to|qr\.net|1url\.com', url)
        return 1 if match else 0

    shortening_flag = shortening_service(url)

    # 9. Count the number of directories in the URL path
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    dir_count = no_of_dir(url)

    # 10. Check for suspicious words in URL (e.g., 'login', 'paypal')
    def suspicious_words(url):
        match = re.search(r'PayPal|login|signin|bank|account|update|free|service|bonus|ebayisapi|webscr', url)
        return 1 if match else 0

    suspicious_flag = suspicious_words(url)

    # 11. Calculate hostname length
    hostname_length = len(urlparse(url).netloc)

    # 12. Count the number of uppercase letters in the URL
    upper_count = sum(1 for char in url if char.isupper())

    # 13. Count the number of lowercase letters in the URL
    lower_count = sum(1 for char in url if char.islower())

    # 14. Check if the URL has a "www" prefix
    has_www = 1 if 'www.' in url else 0

    # 15. Count number of subdomains (split by '.')
    subdomain_count = len(urlparse(url).hostname.split('.')) - 2 if urlparse(url).hostname else 0

    # 16. Count the number of query parameters
    query_count = len(urlparse(url).query.split('&')) if urlparse(url).query else 0

    # 17. Count the number of fragments in the URL
    fragment_count = 1 if urlparse(url).fragment else 0

    # 18. Check if the URL uses a port number
    has_port = 1 if urlparse(url).port else 0

    # 19. Count the number of slashes in the URL
    slash_count = url.count('/')

    # 20. Check if the URL uses a path
    has_path = 1 if urlparse(url).path else 0

    # 21. Check if the URL contains "http"
    contains_http = 1 if 'http' in url else 0

    # 22. Check if the URL contains a valid top-level domain
    valid_tld = 1 if process_tld(url) else 0

    # 23. Check if the URL contains a valid domain (e.g., example.com)
    has_valid_domain = 1 if domain else 0

    # 24. Check if the URL contains the string "secure"
    contains_secure = 1 if 'secure' in url else 0

    # 25. Check if the URL contains the string "login"
    contains_login = 1 if 'login' in url else 0

    # 26. Check if the URL contains the string "signup"
    contains_signup = 1 if 'signup' in url else 0

    # Combine all features into a dictionary
    features_dict = {
        'url_len': url_len,
        '@': feature_counts['@'],
        '?': feature_counts['?'],
        '-': feature_counts['-'],
        '=': feature_counts['='],
        '.': feature_counts['.'],
        '#': feature_counts['#'],
        '%': feature_counts['%'],
        '+': feature_counts['+'],
        '$': feature_counts['$'],
        '!': feature_counts['!'],
        '*': feature_counts['*'],
        ',': feature_counts[','],
        '//': feature_counts['//'],
        'abnormal_url': abnormal_url_flag,
        'https': https_flag,
        'digits': digit_count_value,
        'letters': letter_count_value,
        'Shortening_Service': shortening_flag,
        'count_dir': dir_count,
        'sus_url': suspicious_flag,
        'hostname_length': hostname_length
    }

    # Convert to a DataFrame (for easier handling and saving)
    df_single = pd.DataFrame([features_dict])

    # Save to CSV
    df_single.to_csv('url_test.csv', index=False)

    return df_single



# Route to process the URL and display a message
@app.route('/url', methods=['POST'])
def process_url():
    url = request.form['url_input']
    df_single = process_single_url(url)  # Process the URL here
    # Save the DataFrame or data to CSV (you might want to save it to a file)
    df_single.to_csv('url_test.csv', index=False)

    x=predict_from_csvurl('url_test.csv', "advanced_nn_model.pth",class_names)

    message = f"After prediction this URL is: {x}"
    filename = 'url_test.csv'  # The filename of the saved CSV

    # Render the url.html template and pass the message and filename
    return render_template('url.html', message=message, filename=filename)

############################################################################
#JSON Uploading
# Route to handle the file upload and conversion for JSON
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'json_file' not in request.files:
        return 'No file part in the request.', 400

    file = request.files['json_file']

    if file.filename == '':
        return 'No selected file.', 400

    if file and allowed_json_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print(filename)

        try:
            # Process the JSON file
            with open(filename, 'r') as f:
                data = json.load(f)

            # Normalize the JSON data and convert it to a DataFrame
            df = pd.json_normalize(data)


            # Save the CSV file in the uploads directory with a similar name
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            df.to_csv('cloud_test.csv', index=False)

            # Predict using the processed data
            x = predict_new_data(csv_file_path, loaded_model, loaded_scaler, numerical_cols, categorical_cols)
            if x == 0:
                x="not applicable"
            else:
                x="safe"

            message = f"After prediction, this JSON file is: {x}"
            filename = 'cloud_test.csv'  # The filename of the saved CSV

            # Render the cloud.html template and pass the message and filename
            return render_template('cloud.html', message=message, filename=filename)

        except json.JSONDecodeError as e:
            return f"Error parsing JSON file: {e}", 400
        except Exception as e:
            return f"An error occurred while processing the file: {e}", 500

    return 'Invalid file type. Please upload a JSON file.', 400


############################################################################
#PCAP Uploading
# Route to handle the PCAP file upload
@app.route('/upload-pcap', methods=['POST'])
def upload_pcap():
    if 'pcap_file' not in request.files:
        return 'No file part', 400
    file = request.files['pcap_file']

    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_pcap_file(file.filename):
        # Save the uploaded PCAP file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Convert PCAP file to CSV
        output_file = r'../models/network/network_test.csv'
        pcap_to_csv(filename, 'network_test.csv')
        csv_file = "network_test.csv"  # Replace with your actual CSV file path
        predicted_class = predict_from_netcsv(csv_file)

        # Provide the path to the converted CSV
        return render_template(
            'network.html',
            message=f"PCAP file successfully uploaded and converted to CSV and after prediction the class is: {predicted_class}"
        )

    return 'Invalid file type. Please upload a PCAP file.', 400


# Convert PCAP file to CSV
def pcap_to_csv(input_file, output_file):
    """Convert PCAP file to CSV with custom fields."""

    # Create a new event loop for the current thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Capture the packets
    capture = pyshark.FileCapture(input_file, keep_packets=False)

    headers = [
        'Flow Duration', 'Total Fwd Packets', 'Fwd Packets', 'Bwd Packets',
        'Flow IAT', 'SYN Flag Count', 'FIN Flag Count', 'Active Mean', 'Idle Mean'
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        flow_data = defaultdict(lambda: 0)
        flow_data['timestamps'] = []

        for packet_number, packet in enumerate(capture, start=1):
            extract_custom_fields(packet, flow_data)

            if packet_number % 100 == 0:
                print(f"Processed {packet_number} packets")

        finalize_features(flow_data)
        # Update dynamically if needed
        writer.writerow(flow_data)

    print(f"Conversion complete. Output saved to {output_file}")
def extract_custom_fields(packet, flow_data):
    """Extract custom fields from a packet."""
    try:
        frame_time_delta = float(packet.frame_info.time_delta)
        flow_data['Flow Duration'] += frame_time_delta * 1e6

        if hasattr(packet, 'ip') and hasattr(packet, 'tcp'):
            if 'src' in packet.ip.field_names and 'dst' in packet.ip.field_names:
                if packet.ip.src < packet.ip.dst:
                    flow_data['Total Fwd Packets'] += 1
                    flow_data['Fwd Packets'] += int(packet.tcp.len or 0)
                else:
                    flow_data['Bwd Packets'] += int(packet.tcp.len or 0)

        if hasattr(packet, 'tcp'):
            flow_data['SYN Flag Count'] += int(packet.tcp.flags_syn == '1')
            flow_data['FIN Flag Count'] += int(packet.tcp.flags_fin == '1')

        timestamp = float(packet.frame_info.time_epoch)
        flow_data['timestamps'].append(timestamp)

    except AttributeError:
        pass
def finalize_features(flow_data):
    """Calculate derived features like Active Mean and Idle Mean."""
    timestamps = flow_data['timestamps']

    if len(timestamps) > 1:
        time_differences = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        active_times = [t for t in time_differences if t < 1]
        flow_data['Active Mean'] = sum(active_times) / len(active_times) if active_times else 0

        idle_times = [t for t in time_differences if t >= 1]
        flow_data['Idle Mean'] = sum(idle_times) / len(idle_times) if idle_times else 0
    else:
        flow_data['Active Mean'] = 0
        flow_data['Idle Mean'] = 0

    del flow_data['timestamps']
# Route to allow users to download the converted CSV
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
############################################################################
#Routing with HTML and CSS
# Set up the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_JSON_EXTENSIONS'] = {'json'}
app.config['ALLOWED_PCAP_EXTENSIONS'] = {'pcap', 'pcapng'}


# Function to check if the file extension is allowed for JSON
def allowed_json_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_JSON_EXTENSIONS']


# Function to check if the file extension is allowed for PCAP
def allowed_pcap_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_PCAP_EXTENSIONS']
# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Network route
@app.route('/network')
def network():
    return render_template('network.html')


# Cloud route
@app.route('/cloud')
def cloud():
    return render_template('cloud.html')


# URL route
@app.route('/url', methods=['GET', 'POST'])
def url():
    message = ''
    if request.method == 'POST':
        url_input = request.form['url_input']
        message = f"URL processed: {url_input}"
    return render_template('url.html', message=message)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
