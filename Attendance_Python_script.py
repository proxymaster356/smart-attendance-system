import cv2
import numpy as np
import os
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import face_recognition
import time
import serial # <--- Keep pyserial
import sys

# --- Configuration ---
REFERENCE_FOLDER = "./faceid_storage"
SPREADSHEET_NAME = "Attendance Logs" # <--- Change this to your Google Sheet name
CREDENTIALS_FILE = 'credentials.json'
# FRAME_PROCESS_INTERVAL = 5 # No longer needed in this logic
RECOGNITION_THRESHOLD = 0.55
UNKNOWN_FACE_LABEL = "Unknown"
DRAW_COLOR_KNOWN = (0, 255, 0)
DRAW_COLOR_UNKNOWN = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX
TOP_LEFT_DISPLAY_COLOR = (255, 255, 0)
FACE_DETECTION_TIMEOUT_SECONDS = 10 # <<<--- Time to wait for face after RFID scan

# --- Arduino/Serial Configuration ---
SERIAL_PORT = 'COM5' # <--- *** CHANGE THIS to your Arduino's port ***
BAUD_RATE = 9600   # *** Match your Arduino's Serial.begin() rate ***
SERIAL_TIMEOUT = 0.1 # Read timeout in seconds (non-blocking check)
SERIAL_PREFIX = "Card ID: " # <<<--- The prefix your Arduino sends
SERIAL_BUZZ_COMMAND = b"BUZZ\n" # <<<--- Command to send to Arduino

# --- RFID to Name Mapping ---
# *** Use the exact IDs your Arduino sends (converted to UPPERCASE here) ***
RFID_NAME_MAP = {
    "13326C28": "MNOP",
    "E38ADA26": "QRST",
    "13183A27": "UVWX",
    # Add more mappings as needed
}
# --- End Configuration ---

# --- Helper Functions --- (normalize_name, enhance_image, setup_google_sheets_dynamic, load_reference_faces)
# Keep these exactly as they were in the original script

def normalize_name(name):
    """Removes leading/trailing whitespace and converts to lowercase."""
    if not isinstance(name, str): return ""
    return name.strip().lower()

def enhance_image(image):
    """Basic image enhancement (optional, can be slow)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def setup_google_sheets_dynamic(spreadsheet_name, credentials_file):
    """Sets up Google Sheets connection and returns worksheet for today."""
    # --- This function remains the same ---
    try:
        today = datetime.now()
        todays_sheet_name = today.strftime("%d_%b_%y")
        print(f"Targeting sheet: {todays_sheet_name}")
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        gc = gspread.authorize(credentials)
        try:
            spreadsheet = gc.open(spreadsheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"Spreadsheet '{spreadsheet_name}' not found. Please create it first.")
            return None
        try:
            worksheet = spreadsheet.worksheet(todays_sheet_name)
            print(f"Found existing sheet: '{todays_sheet_name}'")
        except gspread.exceptions.WorksheetNotFound:
            print(f"Worksheet '{todays_sheet_name}' not found. Creating it...")
            worksheet = spreadsheet.add_worksheet(title=todays_sheet_name, rows="1000", cols="6")
            print(f"Worksheet '{todays_sheet_name}' created.")
        header = ["Timestamp", "ID", "Expected Name", "Detected Name", "Match Result", "Confidence"]
        header_range = 'A1:F1'
        try:
            current_header = worksheet.get(header_range, major_dimension='ROWS')
            if not current_header or not current_header[0] or current_header[0] != header: # Check if empty or incorrect
                 print("Setting/Correcting sheet header.")
                 worksheet.update(range_name=header_range, values=[header])
                 print("Sheet header set/verified.")
        except gspread.exceptions.APIError as api_error:
             print(f"API Error checking/updating header: {api_error}")
             if 'exceeds grid limits' in str(api_error):
                 print("Warning: Header range might be invalid if sheet is brand new and empty. Continuing.")
             else:
                return None
        except Exception as e_header:
             print(f"Error checking/updating header: {e_header}")
        return worksheet
    except FileNotFoundError:
        print(f"Error: Credentials file '{credentials_file}' not found.")
        return None
    except Exception as e:
        print(f"Error connecting/setting up Google Sheets: {e}")
        return None


def load_reference_faces(reference_folder):
    """Loads reference face encodings."""
    # --- This function remains the same, BUT ensure filenames match map values ---
    known_face_encodings = []
    known_face_names = []
    successfully_loaded_count = 0
    if not os.path.isdir(reference_folder):
        print(f"Error: Reference folder '{reference_folder}' not found.")
        return known_face_encodings, known_face_names
    print("--- Loading Reference Faces ---")
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(reference_folder, filename)
            name_id = os.path.splitext(filename)[0]
            try:
                ref_img = face_recognition.load_image_file(filepath)
                face_encodings = face_recognition.face_encodings(ref_img, num_jitters=1, model="large")
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name_id) # Store name exactly as derived from filename
                    successfully_loaded_count += 1
                else:
                    print(f"  Warning: No face found in {filename}")
            except Exception as e:
                 print(f"  Error loading {filename}: {e}")
                 pass
    if successfully_loaded_count > 0:
        print(f"Reference images loaded successfully ({successfully_loaded_count} faces found).")
    else:
        print("Warning: No reference faces successfully loaded.")
    print("------------------------------")
    return known_face_encodings, known_face_names

# --- NEW: Function to handle face detection for a triggered event ---
def detect_and_log_face(ser, worksheet, known_face_encodings, known_face_names,
                        triggered_rfid_id, triggered_rfid_expected_name):
    """
    Activates camera, attempts face detection/recognition within a timeout,
    logs the result, and sends buzz command. Returns True if process completed.
    """
    print(">>> detect_and_log_face activated <<<")
    cap = None # Initialize cap to None
    detection_done = False
    log_entry_made = False

    try:
        print("Activating camera...")
        cap = cv2.VideoCapture(0) # <<<--- Activate camera ON DEMAND
        if not cap or not cap.isOpened():
            print("Error: Cannot open webcam!")
            # Log failure?
            return False # Indicate failure to start camera

        start_time = time.time()
        print(f"Camera activated. Waiting for face... (Timeout: {FACE_DETECTION_TIMEOUT_SECONDS}s)")

        last_logged_face_name_display = "Timeout" # Default if no face found
        last_logged_confidence_display = "N/A"
        log_data_to_send = None

        while time.time() - start_time < FACE_DETECTION_TIMEOUT_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame during detection.")
                time.sleep(0.1) # Avoid busy-waiting on error
                continue

            frame_flipped = cv2.flip(frame, 1)
            # --- Optional: Display window during detection ---
            status_text = f"Scan: {triggered_rfid_id} / Exp: {triggered_rfid_expected_name}"
            cv2.putText(frame_flipped, status_text, (10, 30), FONT, 0.6, TOP_LEFT_DISPLAY_COLOR, 1)
            cv2.putText(frame_flipped, "Detecting Face...", (10, 60), FONT, 0.6, TOP_LEFT_DISPLAY_COLOR, 1)
            time_left = FACE_DETECTION_TIMEOUT_SECONDS - (time.time() - start_time)
            cv2.putText(frame_flipped, f"Timeout in: {time_left:.1f}s", (10, 90), FONT, 0.6, TOP_LEFT_DISPLAY_COLOR, 1)
            cv2.imshow('Face Detection Triggered (Press Q in main window to quit)', frame_flipped)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Allow quitting during this phase too
                 print("Quit key pressed during detection phase.")
                 # Should ideally signal main loop to exit
                 # For simplicity here, just break this detection loop
                 return False # Indicate process was interrupted
            # --- End Optional Display ---

            # Process frame for faces (no need to scale down if only processing few frames)
            # rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB) # Use flipped frame if displaying
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Process original frame data

            current_face_locations = face_recognition.face_locations(rgb_frame, model="hog") # Use "cnn" for GPU, "hog" for CPU

            if current_face_locations:
                print(f"Face detected at T+{time.time() - start_time:.2f}s!")
                current_face_encodings = face_recognition.face_encodings(rgb_frame, current_face_locations, num_jitters=1)

                # --- Process the FIRST detected face for simplicity ---
                face_encoding = current_face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_THRESHOLD)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                detected_name_value = UNKNOWN_FACE_LABEL
                match_result_value = '?' # Unknown
                confidence_sheet_value = "N/A"

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        original_matched_name = known_face_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]

                        detected_name_value = original_matched_name
                        confidence_sheet_value = f"{confidence:.2%}"

                        # Compare with expected name
                        if normalize_name(triggered_rfid_expected_name) == normalize_name(detected_name_value):
                            match_result_value = '✓' # Match
                        else:
                            match_result_value = '✗' # Mismatch
                        print(f"Recognition result: {detected_name_value} (Expected: {triggered_rfid_expected_name}) -> {match_result_value}, Conf: {confidence_sheet_value}")
                    else:
                        detected_name_value = UNKNOWN_FACE_LABEL
                        match_result_value = '?' # No known face match
                        confidence_sheet_value = "N/A" # Or maybe show lowest distance?
                        print(f"Recognition result: Unknown (Expected: {triggered_rfid_expected_name})")
                else: # Should not happen if known_face_encodings is not empty, but safety check
                     detected_name_value = UNKNOWN_FACE_LABEL
                     match_result_value = '?'
                     confidence_sheet_value = "N/A"
                     print(f"Recognition result: Unknown (No distances calculated?)")


                # --- Prepare log data ---
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                log_data_to_send = [timestamp, triggered_rfid_id, triggered_rfid_expected_name,
                                    detected_name_value, match_result_value, confidence_sheet_value]
                last_logged_face_name_display = f"{detected_name_value} ({match_result_value})"
                last_logged_confidence_display = confidence_sheet_value
                detection_done = True
                break # Exit the while loop once a face is detected and processed

            # --- No face detected in this frame, continue loop ---

        # --- After the loop (either face found or timeout) ---
        if not detection_done:
            print("Timeout: No face detected within the time limit.")
            # Log timeout event
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            log_data_to_send = [timestamp, triggered_rfid_id, triggered_rfid_expected_name,
                                "No Face Detected", "Timeout", "N/A"]
            last_logged_face_name_display = "Timeout"
            last_logged_confidence_display = "N/A"


        # --- Log the result to Google Sheets ---
        if log_data_to_send and worksheet:
            try:
                worksheet.append_row(log_data_to_send, value_input_option='USER_ENTERED')
                print(f"--> Logged to Sheet '{worksheet.title}'. Result: {log_data_to_send[4]}")
                log_entry_made = True

                # <<<--- SEND BUZZ COMMAND TO ARDUINO --- >>>
                try:
                    ser.write(SERIAL_BUZZ_COMMAND)
                    print(f"Sent '{SERIAL_BUZZ_COMMAND.decode().strip()}' command to Arduino.")
                except serial.SerialException as serial_write_err:
                    print(f"Error sending command to Arduino: {serial_write_err}")
                except Exception as general_write_err:
                     print(f"Unexpected error sending command to Arduino: {general_write_err}")


            except gspread.exceptions.APIError as sheet_api_error:
                print(f"Google Sheets API Error logging data: {sheet_api_error}")
            except Exception as e:
                print(f"Error writing to Google Sheet: {e}")
        elif not worksheet:
             print("Error: Worksheet object is invalid. Cannot log.")

        # --- Return display info for main loop ---
        return True, last_logged_face_name_display, last_logged_confidence_display # Indicate completion

    except Exception as e:
        print(f"An error occurred during face detection process: {e}")
        # Optionally log this error too
        return False, "Error", "N/A" # Indicate failure

    finally:
        # --- Cleanup ---
        if cap and cap.isOpened():
            cap.release()
            print("Camera released.")
        # Close the temporary display window if it was opened
        if cv2.getWindowProperty('Face Detection Triggered (Press Q in main window to quit)', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Face Detection Triggered (Press Q in main window to quit)')
        print(">>> detect_and_log_face finished <<<")


# --- MODIFIED run_face_recognition ---
def run_face_recognition():
    """Runs face recognition, triggered by specific Arduino serial input."""

    worksheet = setup_google_sheets_dynamic(SPREADSHEET_NAME, CREDENTIALS_FILE)
    if worksheet is None:
        print("Google Sheets setup failed. Exiting.")
        return

    ser = None # Initialize ser to None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"Serial connection established on {SERIAL_PORT} at {BAUD_RATE} baud.")
        time.sleep(2)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        print("Check port name, permissions, and Arduino connection.")
        sys.exit(1)
    except Exception as e_ser_setup:
        print(f"Unexpected error setting up serial port: {e_ser_setup}")
        sys.exit(1)


    known_face_encodings, known_face_names = load_reference_faces(REFERENCE_FOLDER)
    if not known_face_names:
        print("No reference faces loaded. Cannot proceed. Exiting.")
        if ser and ser.is_open: ser.close()
        return

    # --- Don't open camera here initially ---
    # cap = cv2.VideoCapture(0) # REMOVED FROM HERE

    print("\nStarting RFID Attendance System...")
    print("Waiting for RFID scan from Arduino...")
    print(f"(Expecting serial lines starting with: '{SERIAL_PREFIX}')")
    print("Press 'q' to quit.")

    # --- State Management ---
    # We don't need explicit state variables like waiting_for_rfid_trigger anymore,
    # the main loop simply waits for serial input.
    last_logged_face_name_display = "None"
    last_logged_confidence_display = "N/A"

    try:
        while True:
            # --- Primary action: Check for Serial Data from Arduino ---
            triggered = False
            if ser.in_waiting > 0:
                try:
                    serial_line_bytes = ser.readline()
                    serial_line = serial_line_bytes.decode('utf-8').strip()

                    if serial_line:
                        print(f"Serial Raw: '{serial_line}'") # Debug

                        if serial_line.startswith(SERIAL_PREFIX):
                            rfid_id_raw = serial_line[len(SERIAL_PREFIX):].strip()
                            rfid_id_processed = rfid_id_raw.upper() # Ensure uppercase match
                            print(f"Extracted RFID: '{rfid_id_processed}'")

                            expected_name = RFID_NAME_MAP.get(rfid_id_processed)

                            if expected_name:
                                print("-" * 30)
                                print(f"RFID TRIGGER RECEIVED")
                                print(f"  ID: {rfid_id_processed}")
                                print(f"  Expected Name: {expected_name}")
                                print("Attempting face detection and logging...")
                                print("-" * 30)

                                # --- Call the dedicated detection function ---
                                success, disp_name, disp_conf = detect_and_log_face(
                                    ser, worksheet, known_face_encodings, known_face_names,
                                    rfid_id_processed, expected_name
                                )
                                # Update display info based on the result
                                last_logged_face_name_display = disp_name
                                last_logged_confidence_display = disp_conf

                                print(f"\nSystem reset. Waiting for next RFID scan...")

                            else:
                                print(f"WARNING: Received RFID tag ID '{rfid_id_processed}' not found in RFID_NAME_MAP. Ignoring.")
                                # Maybe send a different signal to Arduino? (e.g., short error beep) - Future enhancement

                        else:
                            # Handle other potential Arduino messages (like flame sensor alerts)
                            if "Flame detected" in serial_line:
                                print(f"ALERT from Arduino: {serial_line}") # Display flame alert
                            else:
                                print(f"Ignoring non-RFID serial line: '{serial_line}'")

                    # Clear buffer after processing a line might be good practice
                    # ser.reset_input_buffer() # Optional: uncomment if experiencing duplicate reads

                except UnicodeDecodeError:
                    print("Warning: Could not decode serial data (likely noise or incomplete).")
                    # ser.reset_input_buffer() # Clear potentially corrupt data
                except serial.SerialException as se:
                    print(f"Serial Error during read: {se}. Attempting to continue...")
                    time.sleep(1) # Give port time to recover?

            # --- Keep the main loop running - Display Status ---
            # Create a simple black frame for status display when camera is off
            status_frame = np.zeros((150, 600, 3), dtype=np.uint8) # Black background

            display_y = 30
            status_text = f"Status: Waiting for Arduino RFID Scan..."
            cv2.putText(status_frame, status_text, (10, display_y), FONT, 0.5, TOP_LEFT_DISPLAY_COLOR, 1)
            display_y += 30
            text_name = f"Last Logged: {last_logged_face_name_display}"
            text_conf = f"Last Conf: {last_logged_confidence_display}"
            cv2.putText(status_frame, text_name, (10, display_y), FONT, 0.5, TOP_LEFT_DISPLAY_COLOR, 1)
            display_y += 25
            cv2.putText(status_frame, text_conf, (10, display_y), FONT, 0.5, TOP_LEFT_DISPLAY_COLOR, 1)
            display_y += 30
            cv2.putText(status_frame, "Press 'q' to quit", (10, display_y), FONT, 0.5, (200, 200, 200), 1)


            cv2.imshow('RFID Attendance Status (Press Q to quit)', status_frame)

            # --- Process Keyboard Quit ---
            key = cv2.waitKey(50) & 0xFF # Check key press every 50ms
            if key == ord('q'):
                print("Quit key pressed. Exiting.")
                break

            # Small delay to prevent high CPU usage in the main loop
            # time.sleep(0.05) # Already have waitKey(50)

    finally:
        # --- Cleanup ---
        # Camera is released within detect_and_log_face or if error occurs there
        if ser and ser.is_open:
             ser.close()
             print("Serial port closed.")
        cv2.destroyAllWindows()
        print("Resources released.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    run_face_recognition()
