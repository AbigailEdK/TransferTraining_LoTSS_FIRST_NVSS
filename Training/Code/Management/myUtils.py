# my_utils.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shutil

def send_gmail(subject: str, body: str, to_email: str = "abigail.edk@gmail.com"):
    """
    Send an email via Gmail with a subject and body.

    Args:
        subject (str): Subject line of the email.
        body (str): Body text of the email.
        to_email (str): Recipient email address.
    """
    from_email = "abigail.edk@gmail.com"        # Replace with your email
    password = "Ab9Ga9L.edk"        # Replace with Gmail App Password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {to_email} successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def check_folder_existence(folder_path):
    if os.path.exists(folder_path):
        print(f"Folder {folder_path} already exists. Replace? (y/n)")
        choice = input().lower()
        if choice == 'y':
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
            return True
        else:
            return False
    else:
        os.makedirs(folder_path)
        return True
        