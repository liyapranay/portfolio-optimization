import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
api_key = os.getenv('API_KEY')

if not all([username, password, api_key]):
    print("Please set USERNAME, PASSWORD, and API_KEY in .env")
    exit(1)

# Step 1: Login to get OTP
headers = {
    'X-Mirae-Version': '1',
    'Content-Type': 'application/x-www-form-urlencoded',
}

data = {
    'username': username,
    'password': password,
}

response = requests.post('https://api.mstock.trade/openapi/typea/connect/login', headers=headers, data=data)

if response.status_code == 200:
    print("OTP sent to your registered email/phone. Please check and enter the OTP below.")
else:
    print(f"Login failed: {response.status_code} - {response.text}")
    exit(1)

# Step 2: Get OTP from user
otp = input("Enter OTP: ").strip()

# Step 3: Get access token
data = {
    'api_key': api_key,
    'request_token': otp,
    'checksum': 'L',
}

response = requests.post('https://api.mstock.trade/openapi/typea/session/token', headers=headers, data=data)
# print(response.json())
if response.status_code == 200:
    token_data = response.json()
    access_token = token_data.get('data').get('access_token')
    if access_token:
        # Save to .env
        with open('.env', 'r') as f:
            lines = f.readlines()
        with open('.env', 'w') as f:
            for line in lines:
                if line.startswith('ACCESS_TOKEN='):
                    f.write(f"ACCESS_TOKEN={access_token}\n")
                elif line.startswith('TOKEN_TIMESTAMP='):
                    f.write(f"TOKEN_TIMESTAMP={datetime.now().isoformat()}\n")
                else:
                    f.write(line)
            # If not present, add
            if not any('ACCESS_TOKEN=' in line for line in lines):
                f.write(f"ACCESS_TOKEN={access_token}\n")
            if not any('TOKEN_TIMESTAMP=' in line for line in lines):
                f.write(f"TOKEN_TIMESTAMP={datetime.now().isoformat()}\n")
        print("Access token saved to .env")
    else:
        print("Access token not found in response")
else:
    print(f"Token generation failed: {response.status_code} - {response.text}")